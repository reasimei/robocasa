#!/usr/bin/env python3
"""Evaluate Xiaomi stage AdaLN on all RoboCasa composite_seen target tasks.

The normal Xiaomi policy receives the complete task instruction.  The trained
AdaLN branch additionally receives the current simulator-oracle subtask text.
This is an oracle-stage experiment: the target rollout does not expose the
dataset's per-frame annotation, so each task's simulator state advances the
canonical GT stage sequence loaded from the RoboCasa target data.  DeliverStraw
uses an explicit fallback because its 20250813 target data has no GT stage
columns.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gr00t.eval.simulation import MultiStepConfig, SimulationConfig, VideoConfig
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
from robocasa.utils.dataset_registry_utils import get_task_horizon

from scripts.long_horizon_controller.robocasa_adapter import RobocasaVectorEnvAdapter
from scripts.long_horizon_controller.run_composite_seen_eval import DEFAULT_TASK_INSTRUCTIONS
from scripts.long_horizon_stage_adaln.gt_stage_catalog import (
    DEFAULT_TARGET_MANIFEST,
    GTStageSpec,
    load_gt_stage_catalog,
)
from scripts.long_horizon_stage_adaln.model import adapter_training_args
from scripts.long_horizon_stage_adaln.policy_adapter import StageAdaLNXiaomiPolicyAdapter
from scripts.long_horizon_stage_adaln.benefit_policy_adapter import (
    BenefitGatedXiaomiPolicyAdapter,
)


DEFAULT_MODEL_PATH = "/data/zjw/workspace/Isaac-GR00T/expdata/Xiaomi-Robotics-1-RoboCasa365"
DEFAULT_ADAPTER_PATH = (
    "/data/zjw/workspace/Isaac-GR00T/expdata/long_horizon_stage_adaln/"
    "target_composite_adapter_gate_h2_full/checkpoint-9000.pt"
)
def is_cuda_oom(exc: BaseException) -> bool:
    if exc.__class__.__name__ == "OutOfMemoryError":
        return True
    message = str(exc).lower()
    return "cuda out of memory" in message or "out of memory" in message and "cuda" in message


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--adapter-checkpoint", default=DEFAULT_ADAPTER_PATH)
    parser.add_argument(
        "--adapter-variant",
        choices=["legacy", "benefit_gated"],
        default="legacy",
        help="Use the original adapter or the offline-calibrated residual adapter.",
    )
    parser.add_argument(
        "--benefit-gate-config",
        default="",
        help="JSON produced by calibrate_benefit_gate.py.",
    )
    parser.add_argument("--gt-stage-manifest", default=DEFAULT_TARGET_MANIFEST)
    parser.add_argument("--task-set", default="composite_seen")
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--split", choices=["pretrain", "target"], default="target")
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--output-root", default=(
        "/data/zjw/workspace/Isaac-GR00T/expdata/long_horizon_stage_adaln/"
        "composite_seen_xiaomi_oracle_stage_ckpt9000_20eps"
    ))
    parser.add_argument("--n-action-steps", type=int, default=16)
    parser.add_argument("--max-episode-steps", type=int, default=0)
    parser.add_argument("--history-length", type=int, default=4)
    parser.add_argument("--history-interval-steps", type=int, default=2)
    parser.add_argument("--num-diffusion-steps", type=int, default=5)
    parser.add_argument(
        "--stage-condition-format",
        choices=["full", "subtask_only"],
        default="full",
        help="Text condition injected into DiT AdaLN.",
    )
    parser.add_argument(
        "--stage-condition-scale",
        type=float,
        default=1.0,
        help="Scale the trained AdaLN condition; 0.0 reproduces the base Xiaomi DiT.",
    )
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def unwrap_env(adapter: RobocasaVectorEnvAdapter) -> Any:
    current = adapter._env.envs[0]
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "_check_success") and hasattr(current, "sim"):
            return current
        current = getattr(current, "env", None)
    raise RuntimeError("Could not locate the RoboCasa base environment.")


def safe_bool(fn: Callable[[], Any]) -> bool:
    try:
        return bool(fn())
    except Exception:
        return False


def obj_grasped(env: Any, name: str) -> bool:
    import robocasa.utils.object_utils as ou
    return safe_bool(lambda: ou.check_obj_grasped(env, name))


def obj_far(env: Any, name: str) -> bool:
    import robocasa.utils.object_utils as ou
    return safe_bool(lambda: ou.gripper_obj_far(env, name))


def in_receptacle(env: Any, obj: str, receptacle: str) -> bool:
    import robocasa.utils.object_utils as ou
    return safe_bool(lambda: ou.check_obj_in_receptacle(env, obj, receptacle))


def inside_fixture(env: Any, obj: str, fixture: Any) -> bool:
    import robocasa.utils.object_utils as ou
    return safe_bool(lambda: ou.obj_inside_of(env, obj, fixture))


def fixture_open(fixture: Any, env: Any, threshold: float = 0.95) -> bool:
    if fixture is None:
        return False
    # Drawer.is_open() uses the generic joint normalization.  Drawer has a
    # task-specific get_door_state() that is the authoritative 0..1 fraction.
    get_door_state = getattr(fixture, "get_door_state", None)
    if callable(get_door_state):
        try:
            state = get_door_state(env=env)
            return bool(state) and min(float(value) for value in state.values()) >= threshold
        except Exception:
            pass
    for name in ("is_open", "is_opened"):
        method = getattr(fixture, name, None)
        if callable(method):
            for kwargs in ({"env": env}, {}):
                try:
                    return bool(method(**kwargs))
                except TypeError:
                    continue
                except Exception:
                    break
    return False


def fixture_closed(fixture: Any, env: Any) -> bool:
    method = getattr(fixture, "is_closed", None)
    if callable(method):
        for kwargs in ({"env": env}, {}, {"th": 0.05}):
            try:
                return bool(method(**kwargs))
            except TypeError:
                continue
            except Exception:
                break
    return False


def robot_near_fixture(
    env: Any,
    fixture: Any,
    ref_object: str | None = None,
    distance_threshold: float = 0.35,
    orientation_cos_threshold: float = 0.95,
) -> bool:
    if fixture is None:
        return False
    try:
        from robocasa.utils.env_utils import compute_robot_base_placement_pose
        from robosuite.utils import transform_utils as transform

        target_pos, target_ori = compute_robot_base_placement_pose(
            env,
            ref_fixture=fixture,
            ref_object=ref_object,
        )
        robot_id = env.sim.model.body_name2id("mobilebase0_base")
        base_pos = np.asarray(env.sim.data.body_xpos[robot_id], dtype=float)
        position_error = float(np.linalg.norm(np.asarray(target_pos)[:2] - base_pos[:2]))
        base_ori = transform.mat2euler(
            np.asarray(env.sim.data.body_xmat[robot_id]).reshape((3, 3))
        )
        orientation_cos = float(np.cos(float(target_ori[2]) - float(base_ori[2])))
        return bool(
            position_error <= distance_threshold
            and orientation_cos >= orientation_cos_threshold
        )
    except Exception:
        return False


def toaster_state(env: Any) -> tuple[bool, bool, bool]:
    toaster = getattr(env, "toaster", None)
    if toaster is None:
        return False, False, False
    try:
        slot_values = [
            (
                slot_index,
                toaster.get_state(env=env, slot_pair=slot_index),
            )
            for slot_index in range(len(getattr(toaster, "_slot_pairs", [])))
            if toaster.check_slot_contact(env, "obj", slot_pair=slot_index)
        ]
        values = [value for _, value in slot_values]
        if not values:
            # Preserve a conservative fallback for an unusual initialization
            # where the bread has momentarily lost contact with its slot.
            state = toaster.get_state(env=env)
            values = list(state.values()) if isinstance(state, dict) else []
            slot_values = list(enumerate(values))
    except Exception:
        return False, False, False
    turned_on = any(bool(value.get("turned_on", False)) for value in values)
    lever_down = any(float(value.get("lever", 0.0)) <= 0.70 for value in values)
    # A reset toaster can also have its lever up.  RoboCasa marks a completed
    # toast cycle with _cooldown > 0 after the internal timer expires, which
    # distinguishes a genuinely popped lever from the initial pose.
    cooldown = getattr(toaster, "_cooldown", {})
    lever_popped = False
    for slot_index, value in slot_values:
        try:
            cycle_finished = float(cooldown.get(slot_index, 0.0)) > 0.0
        except (AttributeError, TypeError):
            cycle_finished = False
        lever_popped = lever_popped or (
            cycle_finished
            and not bool(value.get("turned_on", False))
            and float(value.get("lever", 0.0)) >= 0.90
        )
    return turned_on, lever_down, lever_popped


def state_for_task(task: str, env: Any) -> dict[str, bool]:
    """Return simulator predicates keyed by canonical semantic stage IDs."""
    if task == "DeliverStraw":
        drawer = getattr(env, "drawer", None)
        straw_grasped = obj_grasped(env, "straw")
        near_counter = robot_near_fixture(
            env,
            getattr(env, "dining_counter", None),
            ref_object="glass_cup",
        )
        return {
            "gt_0": fixture_open(drawer, env),
            "gt_1": straw_grasped,
            "gt_2": straw_grasped and near_counter,
            "gt_3": safe_bool(env._check_success),
        }
    if task == "GetToastedBread":
        turned_on, lever_down, lever_popped = toaster_state(env)
        near_counter = robot_near_fixture(
            env,
            getattr(env, "dining_counter", None),
            ref_object="plate",
        )
        return {
            "gt_0": turned_on,
            "gt_1": lever_popped,
            "gt_2": obj_grasped(env, "obj"),
            "gt_3": obj_grasped(env, "obj") and near_counter,
            "gt_4": safe_bool(env._check_success),
        }
    if task == "KettleBoiling":
        import robocasa.utils.object_utils as ou
        kettle = env.objects["obj"]
        on_stove = safe_bool(lambda: ou.check_obj_fixture_contact(env, "obj", env.stove))
        on_site = False
        burner_on = False
        try:
            kettle_pos = np.asarray(env.sim.data.body_xpos[env.obj_body_id[kettle.name]])[:2]
            for location, site in env.stove.burner_sites.items():
                if site is None:
                    continue
                dist = np.linalg.norm(
                    np.asarray(env.sim.data.get_site_xpos(site.get("name")))[:2] - kettle_pos
                )
                if dist < 0.15:
                    on_site = True
                    burner_on = burner_on or bool(env.stove.is_burner_on(env=env, burner_loc=location))
        except Exception:
            pass
        return {
            "gt_0": obj_grasped(env, "obj"),
            "gt_1": on_stove and on_site and obj_far(env, "obj"),
            "gt_2": safe_bool(env._check_success),
        }
    if task == "LoadDishwasher":
        dish0 = safe_bool(lambda: env.dishwasher.check_rack_contact(env, "dish0"))
        dish1 = safe_bool(lambda: env.dishwasher.check_rack_contact(env, "dish1"))
        return {
            "gt_0": obj_grasped(env, "dish0"),
            "gt_1": dish0,
            "gt_2": obj_grasped(env, "dish1"),
            "gt_3": dish1,
            "gt_4": safe_bool(env._check_success),
        }
    if task == "PackIdenticalLunches":
        v0 = in_receptacle(env, "vegetable0", "tupperware0")
        m0 = in_receptacle(env, "meat0", "tupperware0")
        v1 = in_receptacle(env, "vegetable1", "tupperware1")
        m1 = in_receptacle(env, "meat1", "tupperware1")
        near_counter = robot_near_fixture(env, getattr(env, "counter", None))
        near_fridge = robot_near_fixture(env, getattr(env, "fridge", None))
        return {
            "gt_0": obj_grasped(env, "meat0"),
            "gt_1": near_counter,
            "gt_2": m0 or m1,
            "gt_3": near_fridge,
            "gt_4": obj_grasped(env, "meat1"),
            "gt_5": near_counter,
            "gt_6": m0 or m1,
            "gt_7": near_fridge,
            "gt_8": obj_grasped(env, "vegetable0"),
            "gt_9": near_counter,
            "gt_10": v0 or v1,
            "gt_11": near_fridge,
            "gt_12": obj_grasped(env, "vegetable1"),
            "gt_13": near_counter,
            "gt_14": v0 and v1,
        }
    if task == "PreSoakPan":
        pan = inside_fixture(env, "obj1", env.sink)
        sponge = inside_fixture(env, "obj2", env.sink)
        water = safe_bool(lambda: env.sink.get_handle_state(env=env)["water_on"])
        return {
            "gt_0": obj_grasped(env, "obj1"),
            "gt_1": pan and obj_far(env, "obj1"),
            "gt_2": obj_grasped(env, "obj2"),
            "gt_3": sponge and obj_far(env, "obj2"),
            "gt_4": safe_bool(env._check_success) or water,
        }
    if task == "PrepareCoffee":
        placed = safe_bool(lambda: env.coffee_machine.check_receptacle_placement_for_pouring(env, "obj"))
        turned_on = bool(getattr(env.coffee_machine, "_turned_on", False))
        return {
            "gt_0": obj_grasped(env, "obj"),
            "gt_1": placed and obj_far(env, "obj"),
            "gt_2": safe_bool(env._check_success) or turned_on,
        }
    if task == "RinseSinkBasin":
        state = env.sink.get_handle_state(env=env)
        washed = getattr(env, "washed_loc", [False, False, False])
        return {
            "gt_0": bool(state.get("water_on", False)),
            "gt_1": safe_bool(env._check_success) or all(washed),
        }
    if task == "ScrubCuttingBoard":
        contacts = int(getattr(env, "board_contact_timer", 0)) >= 5
        swept = False
        positions = getattr(env, "board_contact_positions", [])
        if positions:
            points = np.asarray(positions)
            swept = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0))) >= 0.1
        return {"gt_0": obj_grasped(env, "sponge"), "gt_1": contacts and swept}
    if task == "SearingMeat":
        # check_obj_location_on_stove also requires the burner to be on.  The
        # placement stage must finish before the burner is enabled.
        pan_on_stove = safe_bool(
            lambda: env.stove.get_obj_location_on_stove(
                env, "pan", threshold=0.15
            )
            == env.knob
        )
        meat_in_pan = in_receptacle(env, "meat", "pan")
        burner_on = safe_bool(lambda: env.stove.is_burner_on(env=env, burner_loc=env.knob))
        near_stove = robot_near_fixture(env, getattr(env, "stove", None))
        return {
            "gt_0": obj_grasped(env, "pan"),
            "gt_1": near_stove,
            "gt_2": pan_on_stove and obj_far(env, "pan"),
            "gt_3": obj_grasped(env, "meat"),
            "gt_4": meat_in_pan and obj_far(env, "meat"),
            "gt_5": safe_bool(env._check_success) or burner_on,
        }
    if task == "SetUpCuttingStation":
        knife = in_receptacle(env, "knife", "receptacle")
        meat = in_receptacle(env, "meat", "receptacle")
        near_board = robot_near_fixture(env, getattr(env, "counter", None))
        return {
            "gt_0": obj_grasped(env, "knife"),
            "gt_1": near_board,
            "gt_2": knife and obj_far(env, "knife"),
            "gt_3": near_board,
            "gt_4": obj_grasped(env, "meat"),
            "gt_5": near_board,
            "gt_6": safe_bool(env._check_success) or (meat and obj_far(env, "meat")),
        }
    if task == "StackBowlsCabinet":
        b1 = inside_fixture(env, "bowl1", env.cabinet)
        b2 = inside_fixture(env, "bowl2", env.cabinet)
        stacked = in_receptacle(env, "bowl1", "bowl2") or in_receptacle(env, "bowl2", "bowl1")
        return {
            "gt_0": obj_grasped(env, "bowl1"),
            "gt_1": b1,
            "gt_2": obj_grasped(env, "bowl2"),
            "gt_3": safe_bool(env._check_success) or (b1 and b2 and stacked),
        }
    if task == "SteamInMicrowave":
        veg = in_receptacle(env, "vegetable", "bowl")
        bowl = inside_fixture(env, "bowl", env.microwave)
        closed = fixture_closed(env.microwave, env)
        started = bool(env.microwave.get_state().get("turned_on", False))
        near_microwave = robot_near_fixture(env, getattr(env, "microwave", None))
        return {
            "gt_0": obj_grasped(env, "vegetable"),
            "gt_1": veg,
            "gt_2": obj_grasped(env, "bowl"),
            "gt_3": near_microwave,
            "gt_4": bowl,
            "gt_5": closed,
            "gt_6": safe_bool(env._check_success) or started,
        }
    if task == "StirVegetables":
        vegs = in_receptacle(env, "veg1", "pot") and in_receptacle(env, "veg2", "pot")
        spatula = obj_grasped(env, "spatula")
        stirred = int(getattr(env, "success_time", 0)) >= 5
        return {
            "gt_0": obj_grasped(env, "veg1"),
            "gt_1": in_receptacle(env, "veg1", "pot"),
            "gt_2": obj_grasped(env, "veg2"),
            "gt_3": in_receptacle(env, "veg2", "pot"),
            "gt_4": spatula,
            "gt_5": stirred or safe_bool(env._check_success),
        }
    if task == "StoreLeftoversInBowl":
        chicken = in_receptacle(env, "chicken_drumstick", "bowl")
        veg = in_receptacle(env, "vegetable", "bowl")
        bowl_fridge = safe_bool(lambda: env.fridge.check_rack_contact(env, "bowl"))
        near_fridge = robot_near_fixture(env, getattr(env, "fridge", None))
        return {
            "gt_0": obj_grasped(env, "chicken_drumstick"),
            "gt_1": chicken,
            "gt_2": obj_grasped(env, "vegetable"),
            "gt_3": veg,
            "gt_4": obj_grasped(env, "bowl"),
            "gt_5": near_fridge,
            "gt_6": safe_bool(env._check_success) or bowl_fridge,
        }
    if task == "WashLettuce":
        state = env.sink.get_handle_state(env=env)
        washed = int(getattr(env, "washed_time", 0)) >= 25
        lettuce_under_water = safe_bool(
            lambda: env.sink.check_obj_under_water(env, "lettuce")
        )
        return {
            "gt_0": bool(state.get("water_on", False)),
            "gt_1": obj_grasped(env, "lettuce"),
            "gt_2": washed or lettuce_under_water,
        }
    raise KeyError(f"No oracle stage provider for {task}")


def stage_text(stage: GTStageSpec, condition_format: str) -> str:
    return stage.condition_text(condition_format)


def validate_action(action: dict[str, Any], task: str, episode_index: int, step: int) -> None:
    """Fail early with useful diagnostics when a policy emits invalid controls."""
    parts: list[str] = []
    for key, value in action.items():
        array = np.asarray(value, dtype=np.float32)
        if not np.isfinite(array).all():
            bad = np.argwhere(~np.isfinite(array))[0].tolist()
            raise FloatingPointError(
                f"Non-finite action in {task}/episode_{episode_index:03d} "
                f"at step {step}, key={key}, index={bad}, value={array[tuple(bad)]}"
            )
        parts.append(
            f"{key}:shape={array.shape},min={array.min():.4f},"
            f"max={array.max():.4f},mean={array.mean():.4f}"
        )
    if step < 5 * 16:
        print(
            f"[stage-adaln] action diagnostics {task}/episode_{episode_index:03d} "
            f"step={step}: " + " | ".join(parts),
            flush=True,
        )


def episode_task_instruction(base_env: Any, fallback: str) -> tuple[str, str]:
    """Use the current simulator's concrete language whenever it is available."""
    try:
        value = str(base_env.get_ep_meta().get("lang", "")).strip()
        if value:
            return value, "simulator_ep_meta"
    except Exception:
        pass
    return fallback, "default_task_instruction"


def canonical_stage_labels(
    task: str,
    stages: list[GTStageSpec],
    raw_labels: dict[str, bool],
) -> dict[str, bool]:
    return {
        stage.subtask_id: bool(
            raw_labels.get(stage.subtask_id, False)
            or raw_labels.get(_legacy_stage_key(task, stage.index), False)
        )
        for stage in stages
    }


class MonotonicStageTracker:
    """Track Oracle stages without regressing after a completion is observed."""

    def __init__(self, task: str, stages: list[GTStageSpec]) -> None:
        self.task = task
        self.stages = stages
        self.index = 0
        self.completed = [False] * len(stages)
        self.transitions: list[dict[str, Any]] = []

    def _read_labels(self, env: Any) -> dict[str, bool]:
        return canonical_stage_labels(
            self.task,
            self.stages,
            state_for_task(self.task, env),
        )

    def latched_labels(self) -> dict[str, bool]:
        return {
            stage.subtask_id: bool(self.completed[index])
            for index, stage in enumerate(self.stages)
        }

    def current(
        self,
        env: Any,
        step: int,
    ) -> tuple[GTStageSpec, dict[str, bool], dict[str, bool]]:
        raw_labels = self._read_labels(env)
        if self.index < len(self.stages):
            active = self.stages[self.index]
            if raw_labels.get(active.subtask_id, False):
                self.completed[self.index] = True

        # Advance at most one stage per control cycle.  A completed transient
        # state such as water_on or toaster_on remains latched while the next
        # stage is executed, but future predicates cannot skip the active one.
        if (
            self.index < len(self.stages) - 1
            and self.completed[self.index]
        ):
            next_stage = self.stages[self.index + 1]
            self.transitions.append(
                {
                    "step": step,
                    "from": self.stages[self.index].subtask_id,
                    "to": next_stage.subtask_id,
                }
            )
            self.index += 1

        return self.stages[self.index], raw_labels, self.latched_labels()

    def observe_after_action(
        self,
        env: Any,
    ) -> tuple[dict[str, bool], dict[str, bool]]:
        """Latch the active stage after a rollout chunk without advancing."""
        raw_labels = self._read_labels(env)
        if (
            self.index < len(self.stages)
            and raw_labels.get(self.stages[self.index].subtask_id, False)
        ):
            self.completed[self.index] = True
        return raw_labels, self.latched_labels()


def make_env(args: argparse.Namespace, task_name: str, episode_dir: Path) -> RobocasaVectorEnvAdapter:
    max_steps = args.max_episode_steps or get_task_horizon(task_name)
    video_dir = str(episode_dir / "videos") if args.video else None
    indices = np.arange(
        -(args.history_length - 1) * args.history_interval_steps,
        1,
        args.history_interval_steps,
    )
    config = SimulationConfig(
        env_name=f"robocasa/{task_name}",
        split=args.split,
        n_episodes=1,
        n_envs=1,
        video=VideoConfig(video_dir=video_dir),
        multistep=MultiStepConfig(
            video_delta_indices=indices,
            state_delta_indices=indices,
            n_action_steps=args.n_action_steps,
            max_episode_steps=max_steps,
        ),
    )
    return RobocasaVectorEnvAdapter(simulation_config=config)


def run_episode(
    args: argparse.Namespace,
    task: str,
    default_task_instruction: str,
    policy: Any,
    index: int,
) -> dict[str, Any]:
    import torch

    episode_dir = Path(args.output_root) / "evals" / args.split / task / "episodes" / f"episode_{index:03d}"
    result_path = episode_dir / "result.json"
    if result_path.exists() and not args.overwrite:
        return json.loads(result_path.read_text(encoding="utf-8"))
    episode_dir.mkdir(parents=True, exist_ok=True)
    seed = args.seed_base + index
    torch.manual_seed(seed)
    env = None
    started = time.perf_counter()
    trace: list[dict[str, Any]] = []
    try:
        env = make_env(args, task, episode_dir)
        policy.reset()
        observation, _ = env._env.reset(seed=seed)
        base_env = unwrap_env(env)
        task_instruction, task_instruction_source = episode_task_instruction(
            base_env,
            default_task_instruction,
        )
        stages, stage_source = load_gt_stage_catalog(
            task,
            args.gt_stage_manifest,
            task_instruction=task_instruction,
        )
        tracker = MonotonicStageTracker(task, stages)
        steps = 0
        done = False
        while not done and steps < (args.max_episode_steps or get_task_horizon(task)):
            active, raw_labels_before, latched_labels_before = tracker.current(
                base_env,
                steps,
            )
            action, _ = policy.act(
                observation,
                task_instruction,
                stage_text(active, args.stage_condition_format),
                stage_kind=active.stage,
                **(
                    {"atomic_skill": active.atomic_skill}
                    if args.adapter_variant == "benefit_gated"
                    else {}
                ),
            )
            validate_action(action, task, index, steps)
            observation, _, done, _ = env.step(action)
            steps += args.n_action_steps
            raw_labels, latched_labels = tracker.observe_after_action(
                base_env
            )
            trace.append(
                {
                    "step": steps,
                    "subtask_id": active.subtask_id,
                    "labels": latched_labels,
                    "raw_labels": raw_labels,
                    "latched_labels": latched_labels,
                    "labels_before": raw_labels_before,
                    "latched_labels_before": latched_labels_before,
                }
            )
            if safe_bool(base_env._check_success):
                break
        final_success = safe_bool(base_env._check_success)
        result = {
            "task_name": task,
            "episode_index": index,
            "seed": seed,
            "env_success": final_success,
            "steps": steps,
            "final_subtask_index": tracker.index,
            "transitions": tracker.transitions,
            "task_instruction": task_instruction,
            "task_instruction_source": task_instruction_source,
            "stage_source": stage_source,
            "stage_condition_format": args.stage_condition_format,
            "stages": [stage.to_dict() for stage in stages],
            "stage_trace": trace,
            "elapsed_sec": time.perf_counter() - started,
        }
    except Exception as exc:
        if is_cuda_oom(exc):
            print(
                "[stage-adaln] WARNING: CUDA out of memory at "
                f"{task}/episode_{index:03d}; stopping evaluation immediately. "
                "No failure result was written. Free GPU memory and resume with "
                "the same command/output directory.",
                file=sys.stderr,
                flush=True,
            )
            raise RuntimeError(
                f"CUDA OOM at {task}/episode_{index:03d}; evaluation stopped."
            ) from exc
        result = {
            "task_name": task,
            "episode_index": index,
            "seed": seed,
            "env_success": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    finally:
        if env is not None:
            env.close()
        # RoboCasa/robosuite keeps simulator and renderer objects alive for a
        # while after close().  Release episode-local references before the
        # next task creates another simulator, otherwise long evaluations can
        # accumulate CUDA allocations and fail on the second task.
        env = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def _legacy_stage_key(task: str, index: int) -> str:
    """Map a canonical GT stage position to its simulator predicate."""
    return f"gt_{index}"


def main() -> None:
    args = parse_args()
    tasks = args.tasks or list(TASK_SET_REGISTRY[args.task_set])
    missing = [task for task in tasks if task not in DEFAULT_TASK_INSTRUCTIONS]
    if missing:
        raise ValueError(f"Missing full-task instruction for {missing}")
    if not Path(args.adapter_checkpoint).is_file():
        raise FileNotFoundError(args.adapter_checkpoint)
    training_args = adapter_training_args(args.adapter_checkpoint)
    checkpoint_format = training_args.get("stage_condition_format")
    if checkpoint_format and checkpoint_format != args.stage_condition_format:
        raise ValueError(
            "Adapter was trained with "
            f"--stage-condition-format {checkpoint_format!r}, but evaluation "
            f"requested {args.stage_condition_format!r}."
        )
    policy_kwargs = {
        "model_path": args.model_path,
        "adapter_checkpoint": args.adapter_checkpoint,
        "history_length": args.history_length,
        "action_steps": args.n_action_steps,
        "num_diffusion_steps": args.num_diffusion_steps,
    }
    if args.adapter_variant == "benefit_gated":
        if not args.benefit_gate_config:
            raise ValueError(
                "--benefit-gate-config is required with --adapter-variant benefit_gated"
            )
        policy = BenefitGatedXiaomiPolicyAdapter(
            **policy_kwargs,
            gate_config=args.benefit_gate_config,
        )
    else:
        policy = StageAdaLNXiaomiPolicyAdapter(
            **policy_kwargs,
            condition_scale=args.stage_condition_scale,
        )
    all_results: list[dict[str, Any]] = []
    task_summaries: dict[str, Any] = {}
    for task in tasks:
        # Validate the task has a GT catalog or an explicit manual fallback
        # before allocating the model's evaluation time.
        load_gt_stage_catalog(task, args.gt_stage_manifest)
        default_task_instruction = DEFAULT_TASK_INSTRUCTIONS[task]
        rows = []
        for index in range(args.n_episodes):
            print(f"[stage-adaln] {task} episode {index + 1}/{args.n_episodes}", flush=True)
            row = run_episode(
                args,
                task,
                default_task_instruction,
                policy,
                index,
            )
            rows.append(row)
            all_results.append(row)
        successes = [bool(row["env_success"]) for row in rows]
        task_summaries[task] = {
            "n_episodes": len(rows),
            "successes": successes,
            "success_rate": float(np.mean(successes)) if successes else 0.0,
            "stage_sources": sorted(
                {str(row.get("stage_source", "")) for row in rows}
            ),
            "results": rows,
        }
        output = Path(args.output_root) / "evals" / args.split / task / "summary.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(task_summaries[task], indent=2), encoding="utf-8")
    payload = {
        "method": "xiaomi_stage_adaln_oracle",
        "adapter_checkpoint": args.adapter_checkpoint,
        "adapter_training_args": training_args,
        "stage_condition_format": args.stage_condition_format,
        "adapter_variant": args.adapter_variant,
        "benefit_gate_config": args.benefit_gate_config,
        "seed_base": args.seed_base,
        "n_episodes_per_task": args.n_episodes,
        "tasks": task_summaries,
        "num_episodes": len(all_results),
        "success_rate": float(np.mean([bool(row["env_success"]) for row in all_results])) if all_results else 0.0,
    }
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (root / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(json.dumps({"num_episodes": len(all_results), "success_rate": payload["success_rate"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
