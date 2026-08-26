#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import cv2


def event_frame(event: dict[str, Any], n_action_steps: int, steps_per_render: int) -> int:
    return int(round(int(event["step_index"]) * n_action_steps / max(steps_per_render, 1)))


def compact_fast(payload: dict[str, Any]) -> list[str]:
    trigger = payload.get("trigger", "")
    trigger_short = {
        "suspect_transition": "ST",
        "suspect_complete": "SC",
        "suspect_fail": "SF",
        "timeout": "TO",
    }.get(str(trigger), str(trigger)[:2])
    aux_state = payload.get("aux_state")
    aux_confidence = payload.get("aux_confidence")
    score = payload.get("score")
    parts = [f"F {trigger_short}"]
    if score is not None:
        parts.append(f"cc={float(score):.3f}".replace("0.", "."))
    else:
        parts.append("cc=-")
    if aux_state:
        aux_short = {"success": "succ", "retry": "retry", "progress": "prog"}.get(str(aux_state), str(aux_state)[:4])
        if aux_confidence is not None:
            aux_conf = f"{float(aux_confidence):.3f}".replace("0.", ".")
            parts.append(f"aux={aux_short}:{aux_conf}")
        else:
            parts.append(f"aux={aux_short}")
    else:
        parts.append("aux=-")
    return [" ".join(parts)]


def compact_slow(payload: dict[str, Any]) -> list[str]:
    status = payload.get("status", "")
    status_short = {
        "complete": "done",
        "in_progress": "prog",
        "failed": "fail",
    }.get(str(status), str(status)[:4])
    confidence = payload.get("confidence")
    rationale = str(payload.get("rationale", ""))
    recoveries = payload.get("recovery_subtasks") or []
    recovery_mode = str(payload.get("recovery_mode", ""))
    parts = [f"S {status_short}"]
    if confidence is not None:
        parts.append(f"c={float(confidence):.2f}")
    if recovery_mode == "rollback_retry":
        parts.append(f"rollback={int(payload.get('rollback_steps') or 1)}")
    elif recovery_mode == "insert_recovery":
        parts.append("insert_recovery")
    if recoveries:
        parts.append(f"recovery={len(recoveries)}")
    if rationale:
        parts.append(rationale[:34])
    return [" ".join(parts)]


def compact_env(payload: dict[str, Any]) -> list[str]:
    return [f"ENV done | success={bool(payload.get('success'))}"]


def trim_to_width(text: str, max_width: int, font: int, scale: float, thickness: int) -> str:
    if cv2.getTextSize(text, font, scale, thickness)[0][0] <= max_width:
        return text
    suffix = "..."
    trimmed = text
    while trimmed:
        candidate = trimmed + suffix
        if cv2.getTextSize(candidate, font, scale, thickness)[0][0] <= max_width:
            return candidate
        trimmed = trimmed[:-1]
    return suffix


def draw_bar(frame, lines: list[str], position: str, color: tuple[int, int, int]) -> None:
    if not lines:
        return
    height, width = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.34
    thickness = 1
    line_h = 17
    pad = 7
    box_h = len(lines) * line_h + 2 * pad
    x1 = 6
    x2 = width - 6
    y1 = 6 if position == "top" else height - box_h - 6
    y2 = y1 + box_h
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
    max_text_w = x2 - x1 - 2 * pad
    for idx, line in enumerate(lines):
        text = trim_to_width(line, max_text_w, font, scale, thickness)
        y = y1 + pad + 12 + idx * line_h
        cv2.putText(frame, text, (x1 + pad, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_box(frame, lines: list[str], side: str, color: tuple[int, int, int]) -> None:
    """Legacy corner drawer kept for older callers."""
    if not lines:
        return
    height, width = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.30
    thickness = 1
    line_h = 15
    pad = 7
    text_width = 0
    for line in lines:
        (w, _), _ = cv2.getTextSize(line, font, scale, thickness)
        text_width = max(text_width, w)
    max_box_w = max(80, width // 2 - 10)
    box_w = min(max_box_w, text_width + 2 * pad)
    box_h = len(lines) * line_h + 2 * pad
    x1 = 8 if side == "left" else width - box_w - 8
    y1 = height - box_h - 8
    x2 = x1 + box_w
    y2 = y1 + box_h
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
    for idx, line in enumerate(lines):
        y = y1 + pad + 11 + idx * line_h
        cv2.putText(frame, line, (x1 + pad, y), font, scale, color, thickness, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-action-steps", type=int, default=16)
    parser.add_argument("--steps-per-render", type=int, default=2)
    parser.add_argument("--hold-frames", type=int, default=80)
    args = parser.parse_args()

    source_video = Path(args.video)
    output_video = Path(args.output)
    events = json.loads(Path(args.events).read_text(encoding="utf-8"))

    fast_events: list[tuple[int, list[str]]] = []
    slow_events: list[tuple[int, list[str]]] = []
    for event in events:
        frame = event_frame(event, args.n_action_steps, args.steps_per_render)
        event_type = event.get("event_type")
        payload = event.get("payload", {})
        if event_type == "fast_signal":
            fast_events.append((frame, compact_fast(payload)))
        elif event_type == "vlm_decision":
            slow_events.append((frame, compact_slow(payload)))
        elif event_type == "env_done":
            slow_events.append((frame, compact_env(payload)))
        elif event_type == "insert_recovery":
            slow_events.append((frame, [f"SLOW insert recovery={payload.get('num_recovery_subtasks', 0)}"]))
        elif event_type == "rollback_retry":
            slow_events.append(
                (
                    frame,
                    [
                        "SLOW rollback_retry "
                        f"{payload.get('executed_rollback_chunks', 0)}/"
                        f"{payload.get('requested_rollback_chunks', 0)} chunks"
                    ],
                )
            )
        elif event_type == "rollback_unavailable":
            slow_events.append((frame, ["SLOW rollback unavailable; retry not reversed"]))

    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source_video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_video.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps}",
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-profile:v",
        "baseline",
        "-level",
        "3.0",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_video),
    ]
    writer = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert writer.stdin is not None

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            fast_lines: list[str] = []
            slow_lines: list[str] = []
            for start, lines in fast_events:
                if start <= frame_idx < start + args.hold_frames:
                    fast_lines = lines
            for start, lines in slow_events:
                if start <= frame_idx < start + args.hold_frames:
                    slow_lines = lines
            draw_bar(frame, fast_lines, "top", (255, 255, 255))
            draw_bar(frame, slow_lines, "bottom", (0, 255, 255))
            writer.stdin.write(frame.tobytes())
            frame_idx += 1
    finally:
        cap.release()
        writer.stdin.close()
        stderr = writer.stderr.read().decode("utf-8", errors="replace") if writer.stderr else ""
        return_code = writer.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed with code {return_code}:\n{stderr}")


if __name__ == "__main__":
    main()
