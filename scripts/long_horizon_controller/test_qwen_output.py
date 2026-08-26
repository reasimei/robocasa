#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

from ollama import Client
from pydantic import BaseModel

from .schemas import plan_from_dict
from .vlm_verifier import VLM_DECISION_JSON_SCHEMA, format_vlm_prompt


DEFAULT_EVAL_ROOT = Path(
  'expdata/long_horizon_controller/'
  'composite_seen_full_lhc_aux11000_qwen25vl7b_dualview/evals/target'
)

class VLMDecision(BaseModel):
  status: Literal['complete', 'in_progress', 'failed']
  failure_type: Literal[
    'none',
    'wrong_object',
    'wrong_target',
    'wrong_affordance',
    'wrong_pose',
    'object_dropped',
    'unstable_grasp',
    'state_mismatch',
    'blocked',
    'unreachable',
    'unknown',
  ]
  finish_state_satisfied: bool
  next_start_plausible: bool
  rationale: str

def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description='Check whether an Ollama VLM returns a valid JSON decision.'
  )
  parser.add_argument('--model', default='qwen3-vl:8b')
  parser.add_argument('--base-url', default='http://127.0.0.1:11434')
  parser.add_argument(
    '--eval-root',
    type=Path,
    default=DEFAULT_EVAL_ROOT,
  )
  parser.add_argument('--task', default='KettleBoiling')
  parser.add_argument('--episode', default='episode_000')
  parser.add_argument('--subtask-index', type=int, default=0)
  parser.add_argument(
    '--images',
    nargs=4,
    type=Path,
    default=None,
    metavar=('HIST_VIEW1', 'HIST_VIEW2', 'CUR_VIEW1', 'CUR_VIEW2'),
    help='Exactly four images in verifier order. Defaults to the latest two dual-view timesteps.',
  )
  parser.add_argument('--num-predict', type=int, default=256)
  parser.add_argument('--timeout', type=float, default=300.0)
  parser.add_argument(
    '--format-mode',
    choices=('schema', 'json'),
    default='schema',
    help='Use the Pydantic JSON schema or generic JSON mode.',
  )
  parser.add_argument(
    '--show-thinking',
    action='store_true',
    help='Print the thinking field for diagnosing models that put output there.',
  )
  return parser.parse_args()


def resolve_inputs(args: argparse.Namespace) -> tuple[Any, list[Path]]:
  plan_path = args.eval_root / args.task / 'plan.json'
  plan = plan_from_dict(json.loads(plan_path.read_text(encoding='utf-8')))
  if not 0 <= args.subtask_index < len(plan.subtasks):
    raise IndexError(
      f'--subtask-index {args.subtask_index} is outside plan with '
      f'{len(plan.subtasks)} subtasks.'
    )

  frame_dir = args.eval_root / args.task / 'episodes' / args.episode / 'vlm_frames'
  if args.images:
    image_paths = list(args.images)
  else:
    by_step: dict[str, list[Path]] = {}
    for path in sorted(frame_dir.glob('step_*_image_*.png')):
      step_name = path.name.split('_image_', 1)[0]
      by_step.setdefault(step_name, []).append(path)
    dual_view_steps = [
      paths for _, paths in sorted(by_step.items())
      if len(paths) >= 2
    ]
    if len(dual_view_steps) < 2:
      raise FileNotFoundError(
        f'Need two dual-view timesteps under {frame_dir}, found '
        f'{len(dual_view_steps)}.'
      )
    image_paths = dual_view_steps[-2][:2] + dual_view_steps[-1][:2]

  if len(image_paths) != 4:
    raise ValueError(f'Exactly four images are required, got {len(image_paths)}.')
  for path in image_paths:
    if not path.is_file():
      raise FileNotFoundError(f'Image does not exist: {path}')

  current = plan.subtasks[args.subtask_index]
  next_subtask = (
    plan.subtasks[args.subtask_index + 1]
    if args.subtask_index + 1 < len(plan.subtasks)
    else None
  )
  for subtask in (current, next_subtask):
    if subtask is not None:
      setattr(subtask, 'task_instruction', plan.task_instruction)
  setattr(current, 'vlm_image_timestamps_sec', [-1.0, -1.0, 0.0, 0.0])
  setattr(
    current,
    'vlm_image_labels',
    ['agentview_left', 'eye_in_hand', 'agentview_left', 'eye_in_hand'],
  )
  setattr(
    current,
    'controller_context',
    'Offline replay of a saved dual-view observation; evaluate the current subtask only.',
  )
  return (current, next_subtask, plan), image_paths


def response_dict(response: Any) -> dict[str, Any]:
  if hasattr(response, 'model_dump'):
    return response.model_dump()
  if isinstance(response, dict):
    return response
  raise TypeError(f'Unsupported Ollama response type: {type(response)!r}')


def main() -> None:
  args = parse_args()
  (current, next_subtask, plan), image_paths = resolve_inputs(args)
  prompt = format_vlm_prompt(current, next_subtask, num_images=4)

  client = Client(
    host=args.base_url,
    timeout=args.timeout,
    trust_env=False,
  )
  messages = [{
    'role': 'system',
    'content': (
      'You are a robot visual state verifier. '
      'Return exactly one valid JSON object and nothing else. '
      'Do not output analysis, reasoning, markdown, or commentary. '
      'The response must satisfy the requested JSON schema.'
    ),
  }, {
    'role': 'user',
    'content': prompt,
    'images': [str(path.resolve()) for path in image_paths],
  }]
  response = client.chat(
    model=args.model,
    messages=messages,
    think=False,
    format=(
      VLM_DECISION_JSON_SCHEMA
      if args.format_mode == 'schema'
      else 'json'
    ),
    options={'temperature': 0, 'num_predict': args.num_predict},
  )

  raw = response_dict(response)
  message = raw.get('message') or {}
  content = str(message.get('content') or '')
  thinking = str(message.get('thinking') or '')

  # Print the transport-level fields before parsing so an empty content field
  # can be distinguished from invalid JSON and from JSON emitted in thinking.
  print('--- Ollama response metadata ---')
  print(json.dumps({
    key: raw.get(key)
    for key in (
      'model',
      'created_at',
      'done',
      'done_reason',
      'total_duration',
      'load_duration',
      'prompt_eval_count',
      'eval_count',
    )
    if key in raw
  }, ensure_ascii=False, indent=2))
  print(f'content_length={len(content)}')
  print(f'thinking_length={len(thinking)}')
  if args.show_thinking and thinking:
    print('--- thinking ---')
    print(thinking)
  print('--- content ---')
  print(content)

  if not content.strip():
    raise RuntimeError(
      'Ollama returned an empty message.content. '
      'The response was not parsed as JSON. Use --show-thinking to inspect '
      'whether the model placed its answer in message.thinking.'
    )

  try:
    decision = VLMDecision.model_validate_json(content)
  except Exception as exc:
    raise RuntimeError(
      'Ollama message.content was not valid VLMDecision JSON. '
      f'Raw content: {content!r}'
    ) from exc

  print('--- parsed JSON ---')
  print(json.dumps(decision.model_dump(), ensure_ascii=False, indent=2))
  print('--- verifier input ---')
  print(json.dumps({
    'model': args.model,
    'plan_path': str(args.eval_root / args.task / 'plan.json'),
    'task_instruction': plan.task_instruction,
    'subtask_id': current.subtask_id,
    'image_paths': [str(path) for path in image_paths],
    'image_timestamps_sec': [-1.0, -1.0, 0.0, 0.0],
  }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
  main()
