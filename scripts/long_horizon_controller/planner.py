#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .schemas import TaskPlan, plan_from_json_payload


PLANNER_SYSTEM_PROMPT = """You are a robot task planner for long-horizon Robocasa manipulation.
Decompose the user task into executable subtasks for a VLA policy.

Return ONLY valid JSON:
{
  "subtasks": [
    {
      "subtask_id": "subtask_1",
      "instruction": "short VLA language instruction",
      "expected_start_state": "visual/physical state that should hold before starting",
      "expected_finish_state": "visual/physical state that should hold after completion",
      "max_duration_sec": 20.0,
      "notes": "optional"
    }
  ]
}

Guidelines:
- Each subtask should be directly executable by the VLA policy.
- expected_start_state and expected_finish_state must be visually checkable.
- max_duration_sec is used as a timeout for VLM verification.
- Prefer conservative, atomic subtasks over broad instructions.
"""


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        return json.loads(text[start : end + 1])


@dataclass
class OpenAICompatiblePlanner:
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.2
    timeout_sec: float = 60.0

    def __post_init__(self) -> None:
        self.model = self.model or os.environ.get("OPENAI_MODEL")
        self.base_url = (self.base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not self.model:
            raise ValueError("Set OPENAI_MODEL or pass model=... for the task planner.")
        if not self.api_key:
            raise ValueError("Set OPENAI_API_KEY or pass api_key=... for the task planner.")

    def plan(self, task_instruction: str, context: str = "") -> TaskPlan:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Task instruction:\n{task_instruction}\n\n"
                        f"Optional scene/context:\n{context or 'N/A'}"
                    ),
                },
            ],
        }
        request = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(request, timeout=self.timeout_sec) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Planner API failed: HTTP {exc.code}\n{detail}") from exc

        parsed = json.loads(raw)
        content = parsed["choices"][0]["message"]["content"]
        plan_payload = extract_json_object(content)
        return plan_from_json_payload(
            task_instruction=task_instruction,
            payload=plan_payload,
            planner_model=str(self.model),
            raw_response=content,
        )


@dataclass
class OllamaPlanner:
    model: str | None = None
    base_url: str | None = None
    temperature: float = 0.2
    timeout_sec: float = 180.0
    num_predict: int = 512
    num_gpu: int | None = None
    json_mode: bool = True

    def __post_init__(self) -> None:
        self.model = self.model or os.environ.get("OLLAMA_MODEL") or "llama3.1:70b"
        self.base_url = (
            self.base_url or os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434"
        ).rstrip("/")

    def plan(self, task_instruction: str, context: str = "") -> TaskPlan:
        options: dict[str, Any] = {
            "temperature": self.temperature,
            "num_predict": self.num_predict,
        }
        if self.num_gpu is not None:
            options["num_gpu"] = self.num_gpu

        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "options": options,
            "messages": [
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Task instruction:\n{task_instruction}\n\n"
                        f"Optional scene/context:\n{context or 'N/A'}"
                    ),
                },
            ],
        }
        if self.json_mode:
            payload["format"] = "json"

        request = urllib.request.Request(
            url=f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(request, timeout=self.timeout_sec) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama planner failed: HTTP {exc.code}\n{detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Ollama planner failed to reach {self.base_url}. "
                "Start Ollama or set OLLAMA_BASE_URL."
            ) from exc
        except TimeoutError as exc:
            raise RuntimeError(
                f"Ollama planner timed out after {self.timeout_sec} seconds. "
                "Increase --llm-timeout-sec for large local models."
            ) from exc

        parsed = json.loads(raw)
        content = parsed.get("message", {}).get("content", "")
        if not content:
            raise RuntimeError(f"Ollama planner response did not contain message.content: {raw}")
        plan_payload = extract_json_object(content)
        return plan_from_json_payload(
            task_instruction=task_instruction,
            payload=plan_payload,
            planner_model=f"ollama:{self.model}",
            raw_response=content,
        )


@dataclass
class StaticPlanner:
    """Small fallback planner for smoke tests without API access."""

    max_duration_sec: float = 30.0

    def plan(self, task_instruction: str, context: str = "") -> TaskPlan:
        del context
        payload = {
            "subtasks": [
                {
                    "subtask_id": "subtask_1",
                    "instruction": task_instruction,
                    "expected_start_state": "The scene matches the initial task state.",
                    "expected_finish_state": "The requested task outcome is visually satisfied.",
                    "max_duration_sec": self.max_duration_sec,
                    "notes": "Static one-step fallback plan.",
                }
            ]
        }
        return plan_from_json_payload(task_instruction, payload, planner_model="static")
