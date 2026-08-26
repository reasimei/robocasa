#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .schemas import subtask_from_dict
from .vlm_verifier import LocalQwenVLVerifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    args = parser.parse_args()

    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    current_subtask = subtask_from_dict(request["current_subtask"])
    setattr(current_subtask, "task_instruction", request["current_subtask"].get("task_instruction", ""))
    setattr(
        current_subtask,
        "vlm_image_timestamps_sec",
        request["current_subtask"].get("vlm_image_timestamps_sec"),
    )
    setattr(
        current_subtask,
        "vlm_image_labels",
        request["current_subtask"].get("vlm_image_labels"),
    )
    next_payload = request.get("next_subtask")
    next_subtask = subtask_from_dict(next_payload) if next_payload else None

    verifier = LocalQwenVLVerifier(
        model_path=str(request["model_path"]),
        device_map=str(request.get("device_map", "auto")),
        max_new_tokens=int(request.get("max_new_tokens", 512)),
        fallback_conda_env="",
    )
    image_input = request.get("image_paths", request["image_path"])
    decision = verifier.verify(image_input, current_subtask, next_subtask)
    Path(args.response).write_text(
        json.dumps(asdict(decision), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
