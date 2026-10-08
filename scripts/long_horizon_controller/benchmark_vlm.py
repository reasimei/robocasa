#!/usr/bin/env python3
"""Compare visual verifier models on the same saved Robocasa observations.

Cases are JSONL records with task_instruction, current_subtask, optional
next_subtask, images, and optional expected_status.  Without --cases, cases are
assembled from saved controller events and vlm_frames under --eval-root.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import statistics
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.long_horizon_controller.schemas import SubtaskSpec, VLMStatus, plan_from_dict
from scripts.long_horizon_controller.vlm_verifier import (
    VLM_DECISION_JSON_SCHEMA,
    LocalQwenVLVerifier,
    finalize_vlm_decision,
    format_vlm_prompt,
    image_to_base64_png,
    images_from_any,
    strict_decision_from_text,
)


SYSTEM_PROMPT = (
    "You are a JSON-only robot visual verifier. Return exactly one JSON object "
    "that satisfies the requested schema. Do not output analysis or markdown."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        metavar="BACKEND:MODEL",
        help="Use ollama:MODEL, api:MODEL, or local:/path/to/Qwen-VL-checkpoint.",
    )
    parser.add_argument("--cases", type=Path, help="Labeled or unlabeled JSONL case file.")
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=Path(
            "expdata/long_horizon_controller/composite_seen_full_lhc_aux11000_qwen25vl7b_dualview/evals/target"
        ),
        help="Used to discover cases when --cases is omitted.",
    )
    parser.add_argument("--max-cases", type=int, default=20)
    parser.add_argument("--max-images", type=int, default=4)
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Stratified discovery: choose this many task names.",
    )
    parser.add_argument(
        "--cases-per-task",
        type=int,
        default=5,
        help="Stratified discovery: cases per task (default: 5).",
    )
    parser.add_argument("--complete-per-task", type=int, default=1)
    parser.add_argument("--progress-per-task", type=int, default=2)
    parser.add_argument("--retry-per-task", type=int, default=2)
    parser.add_argument("--sampling-seed", type=int, default=0)
    parser.add_argument("--ollama-base-url", default="http://localhost:11434")
    parser.add_argument(
        "--api-base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument(
        "--api-endpoint", choices=("responses", "chat_completions"), default="responses"
    )
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        help=(
            "Optional API reasoning level forwarded as reasoning_effort for Chat Completions "
            "or reasoning.effort for Responses. The provider must support the supplied value."
        ),
    )
    parser.add_argument("--timeout-sec", type=float, default=180.0)
    parser.add_argument(
        "--api-max-retries",
        type=int,
        default=2,
        help="Retry each transient API failure this many times after the initial request.",
    )
    parser.add_argument(
        "--api-retry-delay-sec",
        type=float,
        default=10.0,
        help="Initial retry delay; later attempts use exponential backoff.",
    )
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("expdata/long_horizon_controller/vlm_benchmark/results.json"),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume successful model/case pairs from --output and checkpoint after every case.",
    )
    parser.add_argument(
        "--export-cases", type=Path, help="Write discovered cases as JSONL and exit."
    )
    return parser.parse_args()


def subtask_from_case(data: dict[str, Any], key: str) -> SubtaskSpec | None:
    item = data.get(key)
    if item is None:
        return None
    if not isinstance(item, dict):
        raise ValueError(f"{key} must be an object.")
    return SubtaskSpec(
        instruction=str(item["instruction"]),
        expected_start_state=str(item["expected_start_state"]),
        expected_finish_state=str(item["expected_finish_state"]),
        max_duration_sec=float(item.get("max_duration_sec", 30.0)),
        subtask_id=str(item.get("subtask_id", "")),
        notes=str(item.get("notes", "")),
    )


def read_cases(path: Path) -> list[dict[str, Any]]:
    cases = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not item.get("images"):
            raise ValueError(f"{path}:{line_number} has no images.")
        subtask_from_case(item, "current_subtask")
        cases.append(item)
    return cases


def _reference_status(event: dict[str, Any]) -> str:
    status = str((event.get("payload") or {}).get("status", "")).strip().lower()
    return "retry" if status == "failed" else status


def _stratified_cases(
    cases: list[dict[str, Any]],
    *,
    num_tasks: int,
    cases_per_task: int,
    complete_per_task: int,
    progress_per_task: int,
    retry_per_task: int,
    seed: int,
) -> list[dict[str, Any]]:
    import random

    requested = {
        "complete": complete_per_task,
        "in_progress": progress_per_task,
        "retry": retry_per_task,
    }
    if sum(requested.values()) != cases_per_task:
        raise ValueError(
            "complete/progress/retry counts must sum to --cases-per-task "
            f"({sum(requested.values())} != {cases_per_task})."
        )
    by_task: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        task = str(case["case_id"]).split("/", 1)[0]
        by_task.setdefault(task, []).append(case)
    rng = random.Random(seed)
    for task_cases in by_task.values():
        rng.shuffle(task_cases)
    ranked_tasks = sorted(
        by_task,
        key=lambda task: (
            len(by_task[task]),
            sum(
                by_task[task][i].get("reference_status") == "in_progress"
                for i in range(len(by_task[task]))
            ),
            sum(
                by_task[task][i].get("reference_status") == "retry"
                for i in range(len(by_task[task]))
            ),
        ),
        reverse=True,
    )
    selected_tasks = ranked_tasks[:num_tasks]
    if len(selected_tasks) < num_tasks:
        raise ValueError(
            f"Only {len(selected_tasks)} tasks have saved VLM frames; requested {num_tasks}."
        )

    selected: list[dict[str, Any]] = []
    for task in selected_tasks:
        pool = list(by_task[task])
        used: set[str] = set()
        for bucket, count in requested.items():
            matching = [
                case
                for case in pool
                if case.get("reference_status") == bucket and case["case_id"] not in used
            ]
            fallback = [
                case for case in pool if case["case_id"] not in used and case not in matching
            ]
            chosen = matching[:count] + fallback[: max(0, count - len(matching))]
            if len(chosen) < count:
                raise ValueError(
                    f"Task {task} has only {len(pool)} usable cases; cannot sample {cases_per_task}."
                )
            for case in chosen:
                used.add(case["case_id"])
                case["sampling_bucket"] = bucket
                case["sampling_fallback"] = case.get("reference_status") != bucket
                selected.append(case)
    return selected


def discover_cases(
    eval_root: Path,
    max_cases: int,
    max_images: int,
    *,
    num_tasks: int | None = None,
    cases_per_task: int = 5,
    complete_per_task: int = 1,
    progress_per_task: int = 2,
    retry_per_task: int = 2,
    sampling_seed: int = 0,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for event_path in sorted(eval_root.glob("*/episodes/*/controller_events.json")):
        task_dir = event_path.parents[2]
        plan_path = task_dir / "plan.json"
        frame_dir = event_path.parent / "vlm_frames"
        if not plan_path.exists() or not frame_dir.exists():
            continue
        plan = plan_from_dict(json.loads(plan_path.read_text(encoding="utf-8")))
        events = json.loads(event_path.read_text(encoding="utf-8"))
        for event in events:
            if event.get("event_type") != "vlm_decision":
                continue
            index = int(event.get("subtask_index", -1))
            step = int(event.get("step_index", -1))
            if not 0 <= index < len(plan.subtasks) or step < 0:
                continue
            images = [
                frame_dir / f"step_{step:06d}_image_{image_index:02d}.png"
                for image_index in range(max_images)
                if (frame_dir / f"step_{step:06d}_image_{image_index:02d}.png").exists()
            ]
            if not images:
                continue
            current = plan.subtasks[index]
            next_subtask = plan.subtasks[index + 1] if index + 1 < len(plan.subtasks) else None
            case = {
                "case_id": f"{task_dir.name}/{event_path.parent.name}/step_{step:06d}",
                "task_instruction": plan.task_instruction,
                "current_subtask": vars(current),
                "next_subtask": vars(next_subtask) if next_subtask else None,
                "images": [str(path) for path in images],
                "source_event": str(event_path),
                "reference_status": _reference_status(event),
            }
            cases.append(case)
    if num_tasks is not None:
        return _stratified_cases(
            cases,
            num_tasks=num_tasks,
            cases_per_task=cases_per_task,
            complete_per_task=complete_per_task,
            progress_per_task=progress_per_task,
            retry_per_task=retry_per_task,
            seed=sampling_seed,
        )
    return cases[:max_cases]


def prompt_for_case(case: dict[str, Any]) -> tuple[str, list[Any], bool]:
    current = subtask_from_case(case, "current_subtask")
    if current is None:
        raise ValueError("current_subtask is required.")
    next_subtask = subtask_from_case(case, "next_subtask")
    current.task_instruction = str(case.get("task_instruction", ""))
    image_paths = [Path(path) for path in case["images"]]
    images = images_from_any(image_paths)
    return (
        format_vlm_prompt(current, next_subtask, num_images=len(images)),
        images,
        next_subtask is not None,
    )


def subtasks_for_case(case: dict[str, Any]) -> tuple[SubtaskSpec, SubtaskSpec | None]:
    current = subtask_from_case(case, "current_subtask")
    if current is None:
        raise ValueError("current_subtask is required.")
    next_subtask = subtask_from_case(case, "next_subtask")
    current.task_instruction = str(case.get("task_instruction", ""))
    return current, next_subtask


def request_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout_sec: float,
    *,
    max_retries: int = 0,
    retry_delay_sec: float = 0.0,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    opener = urllib.request.build_opener()
    transient_codes = {429, 500, 502, 503, 504}
    for attempt in range(max(0, max_retries) + 1):
        try:
            with opener.open(request, timeout=timeout_sec) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            error = RuntimeError(f"HTTP {exc.code} from {url}: {detail}")
            retryable = exc.code in transient_codes
        except (urllib.error.URLError, TimeoutError) as exc:
            error = RuntimeError(f"Could not reach {url}: {exc}")
            retryable = True
        if not retryable or attempt >= max_retries:
            raise error
        delay_sec = retry_delay_sec * (2**attempt)
        print(
            f"Transient API failure ({error}); retrying in {delay_sec:.1f}s "
            f"[{attempt + 1}/{max_retries}]",
            flush=True,
        )
        time.sleep(delay_sec)
    raise AssertionError("unreachable")


def ollama_call(
    model: str, images: list[Any], prompt: str, args: argparse.Namespace
) -> tuple[str, dict[str, Any]]:
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "format": VLM_DECISION_JSON_SCHEMA,
        "options": {"temperature": 0.0, "num_predict": args.max_output_tokens},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": prompt,
                "images": [image_to_base64_png(image) for image in images],
            },
        ],
    }
    response = request_json(
        f"{args.ollama_base_url.rstrip('/')}/api/chat",
        payload,
        {"Content-Type": "application/json"},
        args.timeout_sec,
    )
    return str(response.get("message", {}).get("content", "")), response


def response_text(response: dict[str, Any]) -> str:
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    chunks = []
    for output in response.get("output", []) or []:
        for content in output.get("content", []) or []:
            if isinstance(content.get("text"), str):
                chunks.append(content["text"])
    return "\n".join(chunks)


def api_call(
    model: str, images: list[Any], prompt: str, args: argparse.Namespace
) -> tuple[str, dict[str, Any]]:
    if not args.api_key:
        raise ValueError("API model requested: set OPENAI_API_KEY or pass --api-key.")
    parts = [{"type": "input_text", "text": prompt}]
    parts.extend(
        {"type": "input_image", "image_url": "data:image/png;base64," + image_to_base64_png(image)}
        for image in images
    )
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {args.api_key}"}
    base_url = args.api_base_url.rstrip("/")
    if args.api_endpoint == "responses":
        payload = {
            "model": model,
            "instructions": SYSTEM_PROMPT,
            "input": [{"role": "user", "content": parts}],
            "max_output_tokens": args.max_output_tokens,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "vlm_decision",
                    "strict": True,
                    "schema": VLM_DECISION_JSON_SCHEMA,
                }
            },
        }
        if args.reasoning_effort:
            payload["reasoning"] = {"effort": args.reasoning_effort}
        response = request_json(
            f"{base_url}/responses",
            payload,
            headers,
            args.timeout_sec,
            max_retries=args.api_max_retries,
            retry_delay_sec=args.api_retry_delay_sec,
        )
        return response_text(response), response
    content = [{"type": "text", "text": prompt}]
    content.extend(
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64," + image_to_base64_png(image)},
        }
        for image in images
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "max_completion_tokens": args.max_output_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "vlm_decision",
                "strict": True,
                "schema": VLM_DECISION_JSON_SCHEMA,
            },
        },
    }
    if args.reasoning_effort:
        payload["reasoning_effort"] = args.reasoning_effort
    response = request_json(
        f"{base_url}/chat/completions",
        payload,
        headers,
        args.timeout_sec,
        max_retries=args.api_max_retries,
        retry_delay_sec=args.api_retry_delay_sec,
    )
    return str(response.get("choices", [{}])[0].get("message", {}).get("content", "")), response


def local_call(
    verifier: LocalQwenVLVerifier,
    images: list[Any],
    case: dict[str, Any],
) -> tuple[str, dict[str, Any], Any]:
    current, next_subtask = subtasks_for_case(case)
    decision = verifier.verify(images, current, next_subtask)
    return decision.raw_response, {"backend": "local_qwen_vl"}, decision


def usage(response: dict[str, Any]) -> dict[str, Any]:
    return response.get("usage") or {
        key: response[key] for key in ("prompt_eval_count", "eval_count") if key in response
    }


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [row["latency_sec"] for row in rows if row.get("latency_sec") is not None]
    parsed_latencies = [
        row["latency_sec"]
        for row in rows
        if row.get("latency_sec") is not None and row.get("parse_ok") is True
    ]
    steady_latencies = [
        row["latency_sec"]
        for index, row in enumerate(rows)
        if index > 0 and row.get("latency_sec") is not None and row.get("parse_ok") is True
    ]
    parsed = sum(bool(row.get("parse_ok")) for row in rows)
    labels = {item.value for item in VLMStatus}
    labeled = [row for row in rows if row.get("expected_status") in labels]
    result: dict[str, Any] = {
        "cases": len(rows),
        "parse_rate": parsed / len(rows) if rows else 0.0,
        "mean_latency_sec": sum(latencies) / len(latencies) if latencies else None,
        "median_latency_sec": (statistics.median(latencies) if latencies else None),
        "successful_mean_latency_sec": (
            sum(parsed_latencies) / len(parsed_latencies) if parsed_latencies else None
        ),
        "successful_median_latency_sec": (
            statistics.median(parsed_latencies) if parsed_latencies else None
        ),
        "steady_state_mean_latency_sec": (
            sum(steady_latencies) / len(steady_latencies) if steady_latencies else None
        ),
        "first_case_latency_sec": latencies[0] if latencies else None,
        "p95_latency_sec": (
            sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)] if latencies else None
        ),
        "status_counts": dict(Counter(row.get("status", "error") for row in rows)),
        "reference_status_counts": dict(
            Counter(row.get("reference_status", "unknown") for row in rows)
        ),
        "labeled_cases": len(labeled),
    }
    reference_rows = [
        row for row in rows if row.get("parse_ok") and row.get("reference_status") in labels
    ]
    if reference_rows:
        result["reference_agreement"] = sum(
            row["status"] == row["reference_status"] for row in reference_rows
        ) / len(reference_rows)
    if labeled:
        result["accuracy"] = sum(row["status"] == row["expected_status"] for row in labeled) / len(
            labeled
        )
        per_status = {}
        for status in VLMStatus:
            name = status.value
            actual = sum(row["expected_status"] == name for row in labeled)
            if not actual:
                continue
            true_positive = sum(
                row["expected_status"] == name and row["status"] == name for row in labeled
            )
            predicted = sum(row["status"] == name for row in labeled)
            precision = true_positive / predicted if predicted else 0.0
            recall = true_positive / actual
            per_status[name] = {
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            }
        result["per_status"] = per_status
        result["macro_f1"] = sum(item["f1"] for item in per_status.values()) / len(per_status)
    return result


def save_results(path: Path, results: dict[str, Any]) -> None:
    """Atomically replace the checkpoint so interruption cannot corrupt prior rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        json.dump(results, temporary_file, ensure_ascii=False, indent=2)
        temporary_file.write("\n")
        temporary_path = Path(temporary_file.name)
    temporary_path.replace(path)


def load_resume_results(path: Path, case_count: int) -> dict[str, Any]:
    if not path.exists():
        return {"models": {}, "case_count": case_count}
    try:
        results = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Cannot resume: checkpoint is not valid JSON: {path}") from exc
    if not isinstance(results, dict) or not isinstance(results.get("models"), dict):
        raise RuntimeError(f"Cannot resume: unexpected checkpoint format: {path}")
    results["case_count"] = case_count
    return results


def main() -> None:
    args = parse_args()
    cases = (
        read_cases(args.cases)
        if args.cases
        else discover_cases(
            args.eval_root,
            args.max_cases,
            args.max_images,
            num_tasks=args.num_tasks,
            cases_per_task=args.cases_per_task,
            complete_per_task=args.complete_per_task,
            progress_per_task=args.progress_per_task,
            retry_per_task=args.retry_per_task,
            sampling_seed=args.sampling_seed,
        )
    )
    if not cases:
        raise RuntimeError(
            "No cases found. Provide --cases or an --eval-root containing controller_events.json and vlm_frames."
        )
    if args.export_cases:
        args.export_cases.parent.mkdir(parents=True, exist_ok=True)
        args.export_cases.write_text(
            "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases), encoding="utf-8"
        )
        print(f"Exported {len(cases)} cases to {args.export_cases}")
        return
    results = (
        load_resume_results(args.output, len(cases))
        if args.resume
        else {
            "models": {},
            "case_count": len(cases),
        }
    )
    results["request_config"] = {
        "api_endpoint": args.api_endpoint,
        "reasoning_effort": args.reasoning_effort,
        "max_output_tokens": args.max_output_tokens,
        "timeout_sec": args.timeout_sec,
        "api_max_retries": args.api_max_retries,
    }
    for model_spec in args.models:
        backend, separator, model = model_spec.partition(":")
        if not separator or backend not in {"ollama", "api", "local"} or not model:
            raise ValueError(
                f"Invalid model spec {model_spec!r}; use ollama:MODEL, api:MODEL, or local:PATH."
            )
        local_verifier = LocalQwenVLVerifier(model_path=model) if backend == "local" else None
        existing_rows = (
            results.get("models", {}).get(model_spec, {}).get("results", []) if args.resume else []
        )
        completed_by_case_id = {
            str(row.get("case_id")): row
            for row in existing_rows
            if isinstance(row, dict) and row.get("parse_ok") is True
        }
        rows = []
        for index, case in enumerate(cases, start=1):
            case_id = str(case.get("case_id", index))
            completed = completed_by_case_id.get(case_id)
            if completed is not None:
                completed["expected_status"] = case.get("expected_status")
                rows.append(completed)
                print(
                    f"[{model_spec}] {index}/{len(cases)} {case_id}: resumed",
                    flush=True,
                )
                continue
            prompt, images, has_next = prompt_for_case(case)
            started = time.perf_counter()
            raw_response = ""
            try:
                if backend == "ollama":
                    raw_response, response = ollama_call(model, images, prompt, args)
                    decision = strict_decision_from_text(raw_response)
                elif backend == "api":
                    raw_response, response = api_call(model, images, prompt, args)
                    decision = strict_decision_from_text(raw_response)
                else:
                    raw_response, response, decision = local_call(local_verifier, images, case)
                decision = finalize_vlm_decision(decision, has_next_subtask=has_next)
                assert decision is not None
                row = {
                    "case_id": case_id,
                    "expected_status": case.get("expected_status"),
                    "status": decision.status.value,
                    "failure_type": decision.failure_type,
                    "finish_state_satisfied": decision.finish_state_satisfied,
                    "next_start_plausible": decision.next_start_plausible,
                    "rationale": decision.rationale,
                    "parse_ok": True,
                    "usage": usage(response),
                    "raw_response": raw_response,
                }
            except Exception as exc:
                row = {
                    "case_id": case_id,
                    "expected_status": case.get("expected_status"),
                    "status": "error",
                    "parse_ok": False,
                    "error": str(exc),
                    "raw_response": raw_response,
                }
            row["latency_sec"] = time.perf_counter() - started
            row["reference_status"] = case.get("reference_status")
            row["sampling_bucket"] = case.get("sampling_bucket")
            rows.append(row)
            print(
                f"[{model_spec}] {index}/{len(cases)} {row['case_id']}: {row['status']} ({row['latency_sec']:.2f}s)",
                flush=True,
            )
            results["models"][model_spec] = {"summary": metrics(rows), "results": rows}
            save_results(args.output, results)
        results["models"][model_spec] = {"summary": metrics(rows), "results": rows}
        save_results(args.output, results)
    print(
        json.dumps(
            {key: value["summary"] for key, value in results["models"].items()},
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Saved detailed results to {args.output}")


if __name__ == "__main__":
    main()
