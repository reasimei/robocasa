#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from io import BytesIO
import base64
from dataclasses import dataclass
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .schemas import (
    RecoveryMode,
    SubtaskSpec,
    VLMDecision,
    VLMStatus,
    subtask_from_dict,
)


DEFAULT_QWEN3_VL_PATH = (
    "/data/zjw/.cache/huggingface/hub/models--unsloth--Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"
    "/snapshots/b5b904c3fcdc7541adf2a2bb219b0ed95288c794"
)


VLM_PROMPT_TEMPLATE = """You are a robot execution verifier for long-horizon manipulation.

Given the image sequence and task states, decide whether the current subtask is complete,
still progressing, or failed.

## Decision Procedure

First evaluate two booleans independently:

1. finish_state_satisfied:
- Check ONLY `Current expected finish state`.
- True only if required finish state is visibly satisfied.
- Do not use next subtask or overall task goal.

2. next_start_plausible:
- Check ONLY `Next subtask expected start state`.
- True if the required starting condition is visibly satisfied or physically plausible.
- If no next subtask exists, set True.

Then decide:

- complete:
  finish_state_satisfied=True AND next_start_plausible=True

- in_progress:
  finish_state_satisfied=False OR next_start_plausible=False AND the robot is clearly making progress toward it.

- failed:
  finish_state_satisfied=False OR next_start_plausible=False AND progress is not credible or recovery is needed.

Do NOT mark in_progress only because:
- the robot still holds an object,
- more subtasks remain,
- the whole task is unfinished.

Example:
If the finish state is "robot holds the pan" and the robot holds the pan,
the subtask is complete.

## Failure Types

Use:
none | wrong_object | wrong_target | wrong_affordance | wrong_pose |
object_dropped | unstable_grasp | state_mismatch | blocked |
unreachable | unknown

Typical failed cases:
- wrong object/target/affordance
- incorrect gripper pose (Typically, the gripper should remain open when not grasping anything.)
- dropped object
- unstable grasp
- blocked or unreachable motion
- repeated ineffective actions
- inexplicable twitching

## Input

Full task:
{task_instruction}

Current subtask:
{current_instruction}

Current expected start state:
{current_start}

Current expected finish state:
{current_finish}

Next subtask expected start state:
{next_start}

State transition check:
{state_check}

Controller context:
{controller_context}

Image context:
{image_context}

Images:
1. agentview_left, previous timestep
2. eye_in_hand, previous timestep
3. agentview_left, current timestep
4. eye_in_hand, current timestep

Images 1&3 and 2&4 show temporal change.
Images at the same timestep are complementary views.
Use temporal pairs for progress and current views for state verification.

## Output

Return ONLY valid JSON:

{{
    "status":"complete | in_progress | failed",
    "failure_type":"none | wrong_object | wrong_target | wrong_affordance | wrong_pose | object_dropped | unstable_grasp | state_mismatch | blocked | unreachable | unknown",
    "finish_state_satisfied":true | false,
    "next_start_plausible":true | false,
    "rationale":"brief evidence-based explanation"
}}

Constraints:
- Use only visual evidence.
- Keep rationale concise.
- Output JSON only.
"""


RECOVERY_PROMPT_TEMPLATE = """You are an expert robot recovery planner for long-horizon manipulation.
The current subtask has already been determined to be FAILED.
Choose exactly one recovery mode so the controller can retry the current subtask.

## Input

Full task:
{task_instruction}

Current subtask:
{current_instruction}

Current expected finish state:
{current_finish}

Failure type:
{failure_type}

Controller context:
{controller_context}

Image context:
{image_context}

When four images are provided, they are ordered as:
historical agentview_left, historical eye_in_hand, current agentview_left,
current eye_in_hand. The first two are approximately 1.0 second before the
last two. Use both current views to locate the object and target.

---

## Guidance

### Recovery Principle

Your goal is ONLY to restore the missing preconditions required to retry it.
Always recover to the earliest missing precondition.

### Required Binary Choice

Choose exactly one:

1. `rollback_retry`
   - The failure is caused by recent actions or a recent pose.
   - Reversing recent action chunks can return the robot to a usable earlier state.
   - Set `rollback_steps` to the number of recent action chunks to reverse.
   - Set `recovery_subtasks` to [].
   - Use this only when action reversal is likely to be sufficient.

2. `insert_recovery`
   - Reversing actions is insufficient, unsafe, or the object/state must be actively restored.
   - Set `rollback_steps` to 0.
   - Output 1-3 concrete `recovery_subtasks` before retrying the blocked subtask.

Do not output both modes. Do not use `rollback_retry` for a dropped object,
wrong object, wrong target, broken grasp, or a state that requires active
re-grasping or repositioning. In those cases use `insert_recovery`.

---

### Typical Recovery Actions

Prefer actions such as:

- relocate target object
- move to pre-grasp pose
- re-align gripper
- re-grasp object
- reposition end-effector
- clear collision
- return object to a reachable state

Avoid unnecessary actions.
Do not continue later subtasks.

---

### Failure-specific Guidance

wrong_object→ recover by identifying and approaching the correct object.

wrong_target→ recover by returning to the correct target.

wrong_pose→ recover by repositioning.

unstable_grasp→ recover by re-grasping.

object_dropped→ recover by locating and grasping the object again.

blocked→ recover by moving to a collision-free pose.

state_mismatch→ restore the expected start state.

---

## Output Format

Return ONLY valid JSON. Do not include the thought process.

{{
    "recovery_mode":"insert_recovery | rollback_retry",
    "rollback_steps":0,
    "rationale":"brief reason for the selected recovery mode",
    "recovery_subtasks":[
        {{
            "instruction":"short recovery instruction",

            "expected_start_state":"visual start state",

            "expected_finish_state":"visual finish state",

            "max_duration_sec":10.0,

            "notes":"optional"
        }}
    ]
}}

---

## Constraints

- Propose the minimum number of recovery_subtasks needed before retrying the blocked current subtask.
- For `rollback_retry`, return an empty recovery_subtasks list and a positive rollback_steps value.
- For `insert_recovery`, return rollback_steps=0 and at least one recovery_subtask.
- Use concrete visual robot manipulation instructions, not diagnostics.
- Keep every recovery instruction short enough to be used directly as the VLA language input.
- Make the recovery sequence explicit when needed, for example: re-approach, re-grasp, reposition, clear obstruction, then retry.
- If the object is no longer in the expected hand or place, recovery should restore the exact expected start state of the blocked subtask.
- Return JSON only.
"""

def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        last_error: json.JSONDecodeError | None = None
        for match in re.finditer(r"\{", text):
            try:
                parsed, _ = decoder.raw_decode(text[match.start() :])
            except json.JSONDecodeError as exc:
                last_error = exc
                continue
            if isinstance(parsed, dict):
                return parsed
        if last_error is not None:
            raise last_error
        raise


FailureType = Literal[
    "none",
    "wrong_object",
    "wrong_target",
    "wrong_affordance",
    "wrong_pose",
    "object_dropped",
    "unstable_grasp",
    "state_mismatch",
    "blocked",
    "unreachable",
    "unknown",
]


class VLMDecisionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["complete", "in_progress", "failed"]
    failure_type: FailureType
    finish_state_satisfied: bool = Field(
        description=(
            "ONLY whether the current subtask expected finish state is visibly satisfied. "
            "Do not use this for the next subtask."
        )
    )
    next_start_plausible: bool = Field(
        description=(
            "ONLY whether the next subtask expected start state is visibly plausible. "
            "Do not use this for the current subtask finish state."
        )
    )
    rationale: str = Field(description="brief visual reason")


class RecoverySubtaskResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instruction: str
    expected_start_state: str
    expected_finish_state: str
    max_duration_sec: float
    notes: str = ""


class RecoveryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recovery_mode: Literal["insert_recovery", "rollback_retry"]
    rollback_steps: int = Field(default=0, ge=0)
    rationale: str = ""
    recovery_subtasks: list[RecoverySubtaskResponse]


VLM_DECISION_JSON_SCHEMA: dict[str, Any] = VLMDecisionResponse.model_json_schema()
RECOVERY_JSON_SCHEMA: dict[str, Any] = RecoveryResponse.model_json_schema()


def image_from_any(image: Any) -> Image.Image:
    if isinstance(image, (list, tuple)):
        if not image:
            raise ValueError("Cannot convert an empty image sequence.")
        return image_from_any(image[-1])
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    array = np.asarray(image)
    if array.ndim == 4:
        array = array[-1]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def images_from_any(image: Any) -> list[Image.Image]:
    if isinstance(image, (list, tuple)):
        return [image_from_any(item) for item in image]
    return [image_from_any(image)]


def image_to_base64_png(image: Image.Image) -> str:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def image_context_text(
    num_images: int,
    timestamps_sec: list[float] | None = None,
    image_labels: list[str] | None = None,
) -> str:
    if num_images <= 1:
        return (
            "One image is provided. It is the current observation at t=0.0s. "
            "The image data is supplied separately in the message images array."
        )
    if (
        timestamps_sec
        and len(timestamps_sec) == num_images
        and image_labels
        and len(image_labels) == num_images
    ):
        labels = []
        for index, offset in enumerate(timestamps_sec, start=1):
            camera = image_labels[index - 1]
            role = "current" if abs(offset) < 1e-9 else "historical"
            labels.append(
                f"Image {index}: {camera}, {role}, relative time {offset:+.3f}s"
            )
        timing = "\n".join(labels)
        return (
            f"{num_images} images are supplied separately in the message images array. "
            "They are grouped by timestamp; camera views with the same timestamp "
            "are complementary views of the same moment:\n"
            f"{timing}\n"
            "Compare historical and current images for progress, stalling, ineffective "
            "motion, broken grasps, or movement away from the target. Use both current "
            "camera views when checking the finish state."
        )
    if num_images == 4:
        return (
            "Four images are supplied separately in the message images array. "
            "Assume this order unless explicit labels are provided: "
            "Image 1 is agentview_left at about -1.0s; Image 2 is eye_in_hand "
            "at about -1.0s; Image 3 is agentview_left at the current time; "
            "Image 4 is eye_in_hand at the current time. "
            "Images 1 and 2 are complementary views of the same historical moment; "
            "Images 3 and 4 are complementary views of the current moment. "
            "Compare historical versus current pairs for progress, and use both "
            "current views to judge the finish state."
        )
    return (
        f"{num_images} images are supplied separately in the message images array, "
        "in temporal order from oldest to newest. "
        "Image 1 is the oldest observation and the last image is the current observation at t=0.0s. "
        "Compare the sequence for progress, stalling, repeated ineffective motion, broken grasps, "
        "or movement away from the target."
    )


def format_vlm_prompt(
    current_subtask: SubtaskSpec,
    next_subtask: SubtaskSpec | None,
    num_images: int = 1,
) -> str:
    timestamps_sec = getattr(current_subtask, "vlm_image_timestamps_sec", None)
    image_labels = getattr(current_subtask, "vlm_image_labels", None)
    return VLM_PROMPT_TEMPLATE.format(
        task_instruction=getattr(current_subtask, "task_instruction", ""),
        current_instruction=current_subtask.instruction,
        current_start=current_subtask.expected_start_state,
        current_finish=current_subtask.expected_finish_state,
        next_start=next_subtask.expected_start_state if next_subtask else "No next subtask.",
        state_check=_state_check_text(current_subtask, next_subtask),
        controller_context=getattr(current_subtask, "controller_context", "N/A") or "N/A",
        image_context=image_context_text(
            num_images,
            timestamps_sec=timestamps_sec,
            image_labels=image_labels,
        ),
    )


def format_recovery_prompt(
    current_subtask: SubtaskSpec,
    next_subtask: SubtaskSpec | None,
    num_images: int = 1,
    failure_type: str = "unknown",
    rationale: str = "",
) -> str:
    del next_subtask
    timestamps_sec = getattr(current_subtask, "vlm_image_timestamps_sec", None)
    image_labels = getattr(current_subtask, "vlm_image_labels", None)
    return RECOVERY_PROMPT_TEMPLATE.format(
        task_instruction=getattr(current_subtask, "task_instruction", ""),
        current_instruction=current_subtask.instruction,
        current_start=current_subtask.expected_start_state,
        current_finish=current_subtask.expected_finish_state,
        failure_type=failure_type,
        rationale=rationale,
        controller_context=getattr(current_subtask, "controller_context", "N/A") or "N/A",
        image_context=image_context_text(
            num_images,
            timestamps_sec=timestamps_sec,
            image_labels=image_labels,
        ),
    )


def _normalized_state_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _state_check_text(
    current_subtask: SubtaskSpec,
    next_subtask: SubtaskSpec | None,
) -> str:
    current_finish = current_subtask.expected_finish_state.strip()
    if not next_subtask:
        return (
            "Condition A (current finish): check ONLY this condition:\n"
            f"{current_finish}\n"
            "There is no next subtask, so set Condition B "
            "(next start) to true by convention."
        )

    next_start = next_subtask.expected_start_state.strip()
    if _normalized_state_text(current_finish) == _normalized_state_text(next_start):
        return (
            "Condition A (current finish) and Condition B (next start) are "
            "text-identical. Check this shared condition once:\n"
            f"{current_finish}\n"
            "Copy the result to both boolean fields."
        )

    return (
        "Check two independent conditions. Do not combine or substitute them:\n"
        f"Condition A, current finish only: {current_finish}\n"
        f"Condition B, next start only: {next_start}\n"
        "A negative phrase in either condition is a requirement. For example, "
        "'not X' is satisfied when X is absent or false. Do not treat a "
        "negative requirement as a failure.\n"
        "The next subtask's finish state is irrelevant. The controller advances "
        "only when both A and B are true."
    )


def format_compact_vlm_prompt(
    current_subtask: SubtaskSpec,
    next_subtask: SubtaskSpec | None,
    num_images: int = 1,
) -> str:
    image_labels = getattr(current_subtask, "vlm_image_labels", None)
    image_timestamps = getattr(current_subtask, "vlm_image_timestamps_sec", None)
    return f"""Robot verifier. Be brief and exact. Return ONLY one JSON object.
Do not explain your reasoning. No markdown. First char {{, last char }}.

Schema:
{{"status":"complete|in_progress|failed","confidence":0.0,"rationale":"short visual reason","should_advance":false,"recovery_subtasks":[]}}

Full task: {getattr(current_subtask, "task_instruction", "")}
Current subtask: {current_subtask.instruction}
Current subtask expected start state: {current_subtask.expected_start_state}
Current subtask expected finish state: {current_subtask.expected_finish_state}
{_state_check_text(current_subtask, next_subtask)}
Controller context: {getattr(current_subtask, "controller_context", "N/A") or "N/A"}
Images: {image_context_text(num_images, image_timestamps, image_labels)}

Decision:
- complete only if both the current finish state is satisfied and the next start state is plausible.
- failed if the target/object/robot state is wrong, stuck, moved away, grasp is broken, or recovery is needed before retry.
- in_progress if the condition is not yet satisfied but the robot is still credibly moving toward it.
- If failed, include 1-3 short recovery_subtasks usable as robot language instructions.
"""


def format_compact_recovery_prompt(
    current_subtask: SubtaskSpec,
    next_subtask: SubtaskSpec | None,
    num_images: int = 1,
) -> str:
    del next_subtask
    image_labels = getattr(current_subtask, "vlm_image_labels", None)
    image_timestamps = getattr(current_subtask, "vlm_image_timestamps_sec", None)
    return f"""Robot recovery planner. Inspect image(s) and return ONLY one JSON object.
Schema:
{{"recovery_mode":"insert_recovery|rollback_retry","rollback_steps":0,"rationale":"short visual reason","recovery_subtasks":[{{"instruction":"short recovery command","expected_start_state":"visual start","expected_finish_state":"visual finish","max_duration_sec":10.0,"notes":"optional"}}]}}

Full task: {getattr(current_subtask, "task_instruction", "")}
Blocked subtask: {current_subtask.instruction}
Current subtask expected start state: {current_subtask.expected_start_state}
Current subtask expected finish state: {current_subtask.expected_finish_state}
Controller context: {getattr(current_subtask, "controller_context", "N/A") or "N/A"}
Images: {image_context_text(num_images, image_timestamps, image_labels)}

Choose exactly one recovery_mode. For rollback_retry use an empty recovery_subtasks
list and a positive rollback_steps. For insert_recovery use rollback_steps=0 and
concrete recovery steps.
"""


@dataclass
class LocalQwenVLVerifier:
    model_path: str = DEFAULT_QWEN3_VL_PATH
    device_map: str = "auto"
    max_new_tokens: int = 512
    fallback_conda_env: str = "unsloth"

    def __post_init__(self) -> None:
        self._model = None
        self._processor = None
        self._use_subprocess = False
        self._last_model_load_sec = 0.0

    def _load(self) -> None:
        self._last_model_load_sec = 0.0
        if self._model is not None or self._use_subprocess:
            return
        load_start = time.perf_counter()
        try:
            import torch
            from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("Local Qwen-VL verifier requires torch and transformers.") from exc

        try:
            AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        except ValueError as exc:
            try:
                transformers_version = version("transformers")
            except PackageNotFoundError:
                transformers_version = "unknown"
            if self.fallback_conda_env:
                self._use_subprocess = True
                return
            raise RuntimeError(
                "The local VLM checkpoint uses model_type=qwen3_vl, but this environment's "
                f"transformers version ({transformers_version}) does not recognize it. "
                "Upgrade transformers in the robocasa environment to a Qwen3-VL-capable version, "
                "then rerun with --verifier qwen_vl."
            ) from exc

        self._processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            device_map=self.device_map,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self._model.eval()
        self._last_model_load_sec = time.perf_counter() - load_start

    def verify(
        self,
        image: Any,
        current_subtask: SubtaskSpec,
        next_subtask: SubtaskSpec | None = None,
    ) -> VLMDecision:
        images = images_from_any(image)
        verify_start = time.perf_counter()
        self._load()
        if self._use_subprocess:
            return self._verify_subprocess(image, current_subtask, next_subtask)
        assert self._model is not None and self._processor is not None
        timings = {"model_load_sec": float(self._last_model_load_sec)}
        prompt = format_vlm_prompt(
            current_subtask,
            next_subtask,
            num_images=len(images),
        )
        generate_start = time.perf_counter()
        decision = self._generate_decision(images, prompt)
        decision = finalize_vlm_decision(
            decision,
            has_next_subtask=next_subtask is not None,
        )
        timings["verification_generate_sec"] = time.perf_counter() - generate_start
        if decision.status == VLMStatus.FAILED and decision.recovery_mode == RecoveryMode.NONE:
            recovery_prompt = format_recovery_prompt(
                current_subtask,
                next_subtask,
                num_images=len(images),
                failure_type=decision.failure_type,
                rationale=decision.rationale,
            )
            recovery_start = time.perf_counter()
            recovery_decision = self._generate_decision(images, recovery_prompt)
            recovery_decision = finalize_vlm_decision(
                recovery_decision,
                has_next_subtask=next_subtask is not None,
            )
            timings["recovery_generate_sec"] = time.perf_counter() - recovery_start
            if (
                recovery_decision.recovery_mode != RecoveryMode.NONE
                or recovery_decision.recovery_subtasks
            ):
                decision.recovery_mode = recovery_decision.recovery_mode
                decision.rollback_steps = recovery_decision.rollback_steps
                decision.recovery_subtasks = recovery_decision.recovery_subtasks
                decision.rationale = recovery_decision.rationale or decision.rationale
                decision.raw_response = (
                    f"{decision.raw_response}\n\nRECOVERY_RESPONSE:\n"
                    f"{recovery_decision.raw_response}"
                )
        timings["total_verify_sec"] = time.perf_counter() - verify_start
        decision.timings.update(timings)
        return decision

    def _verify_subprocess(
        self,
        image: Any,
        current_subtask: SubtaskSpec,
        next_subtask: SubtaskSpec | None = None,
    ) -> VLMDecision:
        pil_images = images_from_any(image)
        with tempfile.TemporaryDirectory(prefix="qwen_vl_verify_") as tmp:
            tmp_path = Path(tmp)
            image_paths = []
            for index, pil_image in enumerate(pil_images):
                image_path = tmp_path / f"image_{index:02d}.png"
                pil_image.save(image_path)
                image_paths.append(str(image_path))
            request_path = tmp_path / "request.json"
            response_path = tmp_path / "response.json"
            current_payload = asdict(current_subtask)
            current_payload["task_instruction"] = getattr(current_subtask, "task_instruction", "")
            current_payload["vlm_image_timestamps_sec"] = getattr(
                current_subtask, "vlm_image_timestamps_sec", None
            )
            current_payload["vlm_image_labels"] = getattr(
                current_subtask, "vlm_image_labels", None
            )
            request_path.write_text(
                json.dumps(
                    {
                        "image_path": image_paths[0],
                        "image_paths": image_paths,
                        "model_path": self.model_path,
                        "device_map": self.device_map,
                        "max_new_tokens": self.max_new_tokens,
                        "current_subtask": current_payload,
                        "next_subtask": asdict(next_subtask) if next_subtask else None,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            repo_root = Path(__file__).resolve().parents[2]
            fallback_python = Path(f"/data/zjw/anaconda3/envs/{self.fallback_conda_env}/bin/python")
            if fallback_python.exists():
                command = [
                    str(fallback_python),
                    "-m",
                    "scripts.long_horizon_controller.qwen_vl_worker",
                    "--request",
                    str(request_path),
                    "--response",
                    str(response_path),
                ]
            else:
                command = [
                    "conda",
                    "run",
                    "-n",
                    self.fallback_conda_env,
                    "python",
                    "-m",
                    "scripts.long_horizon_controller.qwen_vl_worker",
                    "--request",
                    str(request_path),
                    "--response",
                    str(response_path),
                ]
            subprocess_start = time.perf_counter()
            completed = subprocess.run(
                command,
                cwd=repo_root,
                check=False,
                capture_output=True,
                text=True,
            )
            subprocess_elapsed = time.perf_counter() - subprocess_start
            if completed.returncode != 0:
                raise RuntimeError(
                    "Qwen-VL subprocess verifier failed.\n"
                    f"stdout:\n{completed.stdout}\n"
                    f"stderr:\n{completed.stderr}"
                )
            payload = json.loads(response_path.read_text(encoding="utf-8"))
            decision = decision_from_payload(payload)
            decision = finalize_vlm_decision(
                decision,
                has_next_subtask=next_subtask is not None,
            )
            decision.timings["subprocess_total_sec"] = subprocess_elapsed
            return decision

    def _generate_decision(self, image: Any, prompt: str) -> VLMDecision:
        assert self._model is not None and self._processor is not None
        pil_images = images_from_any(image)
        content = [{"type": "image", "image": pil_image} for pil_image in pil_images]
        content.append({"type": "text", "text": prompt})
        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]
        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._processor(text=[text], images=pil_images, return_tensors="pt")
        inputs = {key: value.to(self._model.device) for key, value in inputs.items()}
        outputs = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated = outputs[:, inputs["input_ids"].shape[-1] :]
        response = self._processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return decision_from_text(response)


@dataclass
class OllamaVLVerifier:
    model: str = "qwen3-vl:8b"
    base_url: str = "http://localhost:11434"
    timeout_sec: float = 120.0
    temperature: float = 0.0
    num_predict: int = 1024
    keep_alive: str = "30m"
    format_mode: str = "schema"
    supports_image_history: bool = True

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        if self.format_mode not in {"schema", "json"}:
            raise ValueError("format_mode must be 'schema' or 'json'.")

    def verify(
        self,
        image: Any,
        current_subtask: SubtaskSpec,
        next_subtask: SubtaskSpec | None = None,
    ) -> VLMDecision:
        images = images_from_any(image)
        prompt = format_vlm_prompt(current_subtask, next_subtask, num_images=len(images))
        verify_start = time.perf_counter()
        response_payload = self._chat(images, prompt, response_format=VLM_DECISION_JSON_SCHEMA)
        decision = self._decision_from_payload(response_payload, require_status=True)
        if decision is None:
            retry_prompt = (
                "You must repair the previous response.\n"
                "Return exactly ONE minified JSON object and nothing else.\n"
                "No markdown. No prose. No analysis.\n"
                "Schema: {\"status\":\"complete|in_progress|failed\","
                "\"failure_type\":\"none|wrong_object|wrong_target|wrong_affordance|wrong_pose|"
                "object_dropped|unstable_grasp|state_mismatch|blocked|unreachable|unknown\","
                "\"finish_state_satisfied\":true,\"next_start_plausible\":true,"
                "\"rationale\":\"brief visual reason\"}\n"
                "Set both booleans from the images first. "
                "If a next subtask exists, status is complete only when BOTH booleans are true. "
                "If there is no next subtask, set next_start_plausible=true and use "
                "finish_state_satisfied. "
                "Use in_progress only when finish_state_satisfied is false and progress is credible.\n\n"
                f"Original verification prompt:\n{prompt}"
            )
            retry_payload = self._chat(
                images,
                retry_prompt,
                response_format=VLM_DECISION_JSON_SCHEMA,
            )
            retry_decision = self._decision_from_payload(retry_payload, require_status=True)
            retry_decision = finalize_vlm_decision(
                retry_decision,
                has_next_subtask=next_subtask is not None,
            )
            if retry_decision is None:
                first_message = response_payload.get("message", {}) or {}
                retry_message = retry_payload.get("message", {}) or {}
                decision = unparseable_vlm_decision(
                    "\n\nRETRY_RESPONSE:\n".join(
                        text
                        for text in (
                            _message_text(first_message),
                            _message_text(retry_message),
                        )
                        if text.strip()
                    )
                )
                response_payload = retry_payload
            else:
                response_payload = retry_payload
                decision = retry_decision
        decision = finalize_vlm_decision(
            decision,
            has_next_subtask=next_subtask is not None,
        )
        decision.timings.update(self._timings_from_response(response_payload))

        if decision.status == VLMStatus.FAILED and decision.recovery_mode == RecoveryMode.NONE:
            recovery_prompt = format_recovery_prompt(
                current_subtask,
                next_subtask,
                num_images=len(images),
                failure_type=decision.failure_type,
                rationale=decision.rationale,
            )
            recovery_start = time.perf_counter()
            recovery_payload = self._chat(
                images,
                recovery_prompt,
                response_format=RECOVERY_JSON_SCHEMA,
            )
            recovery_decision = self._decision_from_payload(recovery_payload, require_status=False)
            decision.timings["recovery_request_sec"] = time.perf_counter() - recovery_start
            decision.timings.update(
                {
                    f"recovery_{key}": value
                    for key, value in self._timings_from_response(recovery_payload).items()
                }
            )
            if (
                recovery_decision is not None
                and (
                    recovery_decision.recovery_mode != RecoveryMode.NONE
                    or recovery_decision.recovery_subtasks
                )
            ):
                decision.recovery_mode = recovery_decision.recovery_mode
                decision.rollback_steps = recovery_decision.rollback_steps
                decision.recovery_subtasks = recovery_decision.recovery_subtasks
                decision.rationale = recovery_decision.rationale or decision.rationale
                decision.raw_response = (
                    f"{decision.raw_response}\n\nRECOVERY_RESPONSE:\n"
                    f"{recovery_decision.raw_response}"
                )

        decision.timings["total_verify_sec"] = time.perf_counter() - verify_start
        return decision

    @staticmethod
    def _decision_from_payload(
        payload: dict[str, Any],
        require_status: bool = True,
    ) -> VLMDecision | None:
        message = payload.get("message", {}) or {}
        candidates = [
            str(message.get("content", "")),
        ]
        for candidate in candidates:
            if not candidate.strip():
                continue
            try:
                return strict_decision_from_text(candidate, require_status=require_status)
            except (json.JSONDecodeError, ValidationError):
                continue
        return None

    def _chat(
        self,
        images: list[Image.Image],
        prompt: str,
        response_format: Any = VLM_DECISION_JSON_SCHEMA,
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "stream": False,
            "format": response_format if self.format_mode == "schema" else "json",
            "think": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
            },
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a JSON-only robot visual verifier. "
                        "Return exactly one valid JSON object and nothing else. "
                        "Do not output analysis, reasoning, markdown, or commentary. "
                        "The JSON must satisfy the requested response format."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_to_base64_png(image) for image in images],
                }
            ],
        }
        request = urllib.request.Request(
            url=f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(request, timeout=self.timeout_sec) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama VLM verifier failed: HTTP {exc.code}\n{detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Ollama VLM verifier failed to reach {self.base_url}. "
                "Start Ollama or set --ollama-base-url."
            ) from exc
        except TimeoutError as exc:
            raise RuntimeError(
                f"Ollama VLM verifier timed out after {self.timeout_sec} seconds."
            ) from exc

    @staticmethod
    def _timings_from_response(payload: dict[str, Any]) -> dict[str, float]:
        timings: dict[str, float] = {}
        for key in ("total_duration", "load_duration", "prompt_eval_duration", "eval_duration"):
            value = payload.get(key)
            if value is not None:
                timings[f"ollama_{key}_sec"] = float(value) / 1_000_000_000.0
        for key in ("prompt_eval_count", "eval_count"):
            value = payload.get(key)
            if value is not None:
                timings[f"ollama_{key}"] = float(value)
        return timings


def _message_text(message: dict[str, Any]) -> str:
    return "\n\n".join(
        str(message.get(key, ""))
        for key in ("content", "thinking")
        if str(message.get(key, "")).strip()
    )


def decision_from_text(response: str, require_status: bool = True) -> VLMDecision:
    try:
        return strict_decision_from_text(response, require_status=require_status)
    except (json.JSONDecodeError, ValidationError):
        pass
    return decision_from_payload(
        extract_json_object(response),
        raw_response=response,
        require_status=require_status,
    )


def strict_decision_from_text(response: str, require_status: bool = True) -> VLMDecision:
    try:
        if require_status:
            payload = VLMDecisionResponse.model_validate_json(response).model_dump()
        else:
            payload = RecoveryResponse.model_validate_json(response).model_dump()
    except ValidationError as exc:
        raise json.JSONDecodeError(str(exc), response, 0) from exc
    return decision_from_payload(
        payload,
        raw_response=response,
        require_status=require_status,
    )


def loose_decision_from_text(response: str) -> VLMDecision | None:
    text = response.strip()
    if not text:
        return None
    lower = text.lower()
    failed_terms = (
        "status should be failed",
        "status is failed",
        '"status":"failed"',
        '"status": "failed"',
        "therefore failed",
        "use failed",
        "needs recovery",
        "recovery is needed",
        "must recover",
        "stuck",
        "moved away",
        "cannot complete",
        "not on a credible path",
        "broken grasp",
        "dropped",
    )
    complete_terms = (
        "status should be complete",
        "status is complete",
        "finish state is visually satisfied",
        "expected finish is visually satisfied",
        "visually satisfied",
        "should advance",
    )
    progress_terms = (
        "in_progress",
        "in progress",
        "still making",
        "still moving",
        "credible progress",
    )
    if any(term in lower for term in failed_terms):
        status = VLMStatus.FAILED
    elif any(term in lower for term in complete_terms):
        status = VLMStatus.COMPLETE
    elif any(term in lower for term in progress_terms):
        status = VLMStatus.IN_PROGRESS
    else:
        return None
    rationale = "loose parse from Ollama text: " + " ".join(text.split())[:240]
    return VLMDecision(
        status=status,
        confidence=0.0,
        rationale=rationale,
        should_advance=status == VLMStatus.COMPLETE,
        raw_response=text,
    )


def decision_from_payload(
    payload: dict[str, Any],
    raw_response: str = "",
    require_status: bool = True,
) -> VLMDecision:
    if require_status and "status" not in payload:
        raise json.JSONDecodeError("Verifier JSON is missing status.", raw_response or str(payload), 0)
    recovery = []
    for idx, item in enumerate(payload.get("recovery_subtasks", []) or []):
        if isinstance(item, dict):
            recovery.append(subtask_from_dict(item, fallback_id=f"recovery_{idx + 1}"))
        else:
            recovery.append(
                subtask_from_dict(
                    {
                        "instruction": str(item),
                        "expected_start_state": "current visual state",
                        "expected_finish_state": "recovery instruction completed",
                        "max_duration_sec": 10.0,
                    },
                    fallback_id=f"recovery_{idx + 1}",
                )
            )
    if "status" in payload:
        status = parse_vlm_status(payload["status"])
    elif recovery or payload.get("recovery_mode") in {
        RecoveryMode.INSERT_RECOVERY.value,
        RecoveryMode.ROLLBACK_RETRY.value,
    }:
        status = VLMStatus.FAILED
    else:
        status = VLMStatus.IN_PROGRESS
    raw_timings = payload.get("timings", {}) or {}
    timings = {str(key): float(value) for key, value in raw_timings.items()}
    confidence, confidence_label = parse_vlm_confidence(payload.get("confidence", 0.0))
    finish_state_satisfied = payload.get("finish_state_satisfied")
    if not isinstance(finish_state_satisfied, bool):
        finish_state_satisfied = None
    next_start_plausible = payload.get("next_start_plausible")
    if not isinstance(next_start_plausible, bool):
        next_start_plausible = None
    failure_type = str(
        payload.get("failure_type", "none" if status != VLMStatus.FAILED else "unknown")
    ).strip()
    if status != VLMStatus.FAILED:
        failure_type = "none"
    raw_recovery_mode = str(
        payload.get("recovery_mode", RecoveryMode.NONE.value)
    ).strip().lower()
    try:
        recovery_mode = RecoveryMode(raw_recovery_mode)
    except ValueError:
        recovery_mode = RecoveryMode.INSERT_RECOVERY if recovery else RecoveryMode.NONE
    try:
        rollback_steps = max(0, int(payload.get("rollback_steps", 0)))
    except (TypeError, ValueError):
        rollback_steps = 0
    if recovery_mode == RecoveryMode.ROLLBACK_RETRY:
        # A malformed zero is still recoverable: use one recent action chunk,
        # which is the smallest rollback the controller can execute.
        rollback_steps = max(1, rollback_steps)
        recovery = []
    elif recovery_mode == RecoveryMode.INSERT_RECOVERY:
        rollback_steps = 0
        if not recovery:
            recovery_mode = RecoveryMode.NONE
    else:
        rollback_steps = 0
    return VLMDecision(
        status=status,
        confidence=confidence,
        rationale=str(payload.get("rationale", "")),
        finish_state_satisfied=finish_state_satisfied,
        next_start_plausible=next_start_plausible,
        confidence_label=confidence_label,
        failure_type=failure_type,
        recovery_mode=recovery_mode,
        rollback_steps=rollback_steps,
        recovery_subtasks=recovery,
        should_advance=status == VLMStatus.COMPLETE,
        raw_response=str(payload.get("raw_response", raw_response)),
        timings=timings,
    )


def finalize_vlm_decision(
    decision: VLMDecision | None,
    *,
    has_next_subtask: bool,
) -> VLMDecision | None:
    """Apply the controller's two-state completion contract."""
    if decision is None:
        return None

    if not has_next_subtask:
        next_start_plausible = True
        decision.next_start_plausible = True
    else:
        next_start_plausible = decision.next_start_plausible is True

    completion_ready = decision.finish_state_satisfied is True and next_start_plausible
    if completion_ready:
        decision.status = VLMStatus.COMPLETE
        decision.failure_type = "none"
        decision.should_advance = True
    elif decision.status == VLMStatus.COMPLETE:
        # A textual "complete" must not override contradictory state booleans.
        decision.status = VLMStatus.IN_PROGRESS
        decision.failure_type = "none"
        decision.should_advance = False
    else:
        decision.should_advance = False
    return decision


def parse_vlm_status(value: Any) -> VLMStatus:
    text = str(value).strip().lower()
    aliases = {
        "complete": VLMStatus.COMPLETE,
        "completed": VLMStatus.COMPLETE,
        "success": VLMStatus.COMPLETE,
        "succeeded": VLMStatus.COMPLETE,
        "in_progress": VLMStatus.IN_PROGRESS,
        "in progress": VLMStatus.IN_PROGRESS,
        "progress": VLMStatus.IN_PROGRESS,
        "ongoing": VLMStatus.IN_PROGRESS,
        "failed": VLMStatus.FAILED,
        "failure": VLMStatus.FAILED,
        "retry": VLMStatus.FAILED,
    }
    if text in aliases:
        return aliases[text]
    raise json.JSONDecodeError(f"Unsupported VLM status: {value!r}", str(value), 0)


def unparseable_vlm_decision(raw_response: str) -> VLMDecision:
    return VLMDecision(
        status=VLMStatus.IN_PROGRESS,
        confidence=0.0,
        rationale="Ollama returned unparseable JSON; defaulting to in_progress.",
        failure_type="none",
        should_advance=False,
        raw_response=raw_response,
    )


def parse_vlm_confidence(value: Any) -> tuple[float, str]:
    if isinstance(value, str):
        label = value.strip().lower()
        if label == "high":
            return 0.9, label
        if label == "medium":
            return 0.5, label
        if label == "low":
            return 0.2, label
        try:
            return float(label), label
        except ValueError:
            return 0.0, label
    if value is None:
        return 0.0, ""
    try:
        return float(value), ""
    except (TypeError, ValueError):
        return 0.0, str(value)


@dataclass
class DryRunVerifier:
    """Verifier used for controller smoke tests without loading VLM weights."""

    default_status: VLMStatus = VLMStatus.IN_PROGRESS

    def verify(
        self,
        image: Any,
        current_subtask: SubtaskSpec,
        next_subtask: SubtaskSpec | None = None,
    ) -> VLMDecision:
        del image, current_subtask, next_subtask
        return VLMDecision(
            status=self.default_status,
            confidence=0.5,
            rationale="Dry-run verifier.",
            should_advance=self.default_status == VLMStatus.COMPLETE,
            timings={"total_verify_sec": 0.0},
        )
