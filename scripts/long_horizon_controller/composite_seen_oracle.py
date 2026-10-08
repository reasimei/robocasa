#!/usr/bin/env python3
"""Oracle stage catalog and simulator predicates for RoboCasa composite_seen."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from scripts.long_horizon_stage_adaln.gt_stage_catalog import (
    DEFAULT_TARGET_MANIFEST,
    GTStageSpec,
    load_gt_stage_catalog,
)


COMPOSITE_SEEN_TASKS = (
    "DeliverStraw",
    "GetToastedBread",
    "KettleBoiling",
    "LoadDishwasher",
    "PackIdenticalLunches",
    "PreSoakPan",
    "PrepareCoffee",
    "RinseSinkBasin",
    "ScrubCuttingBoard",
    "SearingMeat",
    "SetUpCuttingStation",
    "StackBowlsCabinet",
    "SteamInMicrowave",
    "StirVegetables",
    "StoreLeftoversInBowl",
    "WashLettuce",
)


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


def in_receptacle(env: Any, obj: str, receptacle: str, th: float | None = None) -> bool:
    import robocasa.utils.object_utils as ou

    if th is None:
        return safe_bool(lambda: ou.check_obj_in_receptacle(env, obj, receptacle))
    return safe_bool(lambda: ou.check_obj_in_receptacle(env, obj, receptacle, th=th))


def inside_fixture(env: Any, obj: str, fixture: Any) -> bool:
    import robocasa.utils.object_utils as ou

    return safe_bool(lambda: ou.obj_inside_of(env, obj, fixture))


def fixture_open(fixture: Any, env: Any, threshold: float = 0.95) -> bool:
    if fixture is None:
        return False
    get_door_state = getattr(fixture, "get_door_state", None)
    if callable(get_door_state):
        try:
            state = get_door_state(env=env)
            return bool(state) and min(float(value) for value in state.values()) >= threshold
        except Exception:
            pass
    for name in ("is_open", "is_opened"):
        method = getattr(fixture, name, None)
        if not callable(method):
            continue
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
    if not callable(method):
        return False
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
    """Use RoboCasa's mobile-base placement target and pose tolerances."""
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


def _toaster_state(env: Any) -> tuple[bool, bool, bool]:
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
            state = toaster.get_state(env=env)
            values = list(state.values()) if isinstance(state, dict) else []
            slot_values = list(enumerate(values))
    except Exception:
        return False, False, False

    turned_on = any(bool(value.get("turned_on", False)) for value in values)
    lever_down = any(float(value.get("lever", 0.0)) <= 0.70 for value in values)
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


def _kettle_on_burner(env: Any) -> tuple[bool, str | None, float | None]:
    import robocasa.utils.object_utils as ou

    kettle = env.objects["obj"]
    if not safe_bool(lambda: ou.check_obj_fixture_contact(env, "obj", env.stove)):
        return False, None, None
    kettle_pos = np.asarray(env.sim.data.body_xpos[env.obj_body_id[kettle.name]])[:2]
    closest_location: str | None = None
    closest_distance: float | None = None
    for location, site in env.stove.burner_sites.items():
        if site is None:
            continue
        burner_pos = np.asarray(env.sim.data.get_site_xpos(site.get("name")))[:2]
        distance = float(np.linalg.norm(burner_pos - kettle_pos))
        if closest_distance is None or distance < closest_distance:
            closest_location = location
            closest_distance = distance
    return (
        bool(closest_distance is not None and closest_distance < 0.15),
        closest_location,
        closest_distance,
    )


def _stove_target_object_on_burner(env: Any, object_name: str) -> tuple[bool, str | None, float | None]:
    import robocasa.utils.object_utils as ou

    if not safe_bool(lambda: ou.check_obj_fixture_contact(env, object_name, env.stove)):
        return False, None, None
    obj = env.objects[object_name]
    obj_pos = np.asarray(env.sim.data.body_xpos[env.obj_body_id[obj.name]])[:2]
    closest_location: str | None = None
    closest_distance: float | None = None
    for location, site in env.stove.burner_sites.items():
        if site is None:
            continue
        site_pos = np.asarray(env.sim.data.get_site_xpos(site.get("name")))[:2]
        distance = float(np.linalg.norm(site_pos - obj_pos))
        if closest_distance is None or distance < closest_distance:
            closest_location = location
            closest_distance = distance
    target_location = getattr(env, "knob", None)
    on_target = (
        closest_distance is not None
        and closest_distance < 0.15
        and (target_location is None or closest_location == target_location)
    )
    return bool(on_target), closest_location, closest_distance


def _labels_for_task(task: str, env: Any) -> tuple[list[bool], dict[str, Any]]:
    """Return ordered monotonic-friendly stage predicates and diagnostics."""
    if task == "DeliverStraw":
        drawer = getattr(env, "drawer", None)
        straw_grasped = obj_grasped(env, "straw")
        near_counter = robot_near_fixture(
            env,
            getattr(env, "dining_counter", None),
            ref_object="glass_cup",
        )
        return [
            fixture_open(drawer, env),
            straw_grasped,
            straw_grasped and near_counter,
            safe_bool(env._check_success),
        ], {
            "drawer_open": fixture_open(drawer, env),
            "straw_grasped": straw_grasped,
            "near_dining_counter": near_counter,
        }

    if task == "GetToastedBread":
        turned_on, lever_down, lever_popped = _toaster_state(env)
        near_counter = robot_near_fixture(
            env,
            getattr(env, "dining_counter", None),
            ref_object="plate",
        )
        return [
            turned_on,
            lever_popped,
            obj_grasped(env, "obj"),
            obj_grasped(env, "obj") and near_counter,
            safe_bool(env._check_success),
        ], {
            "toaster_turned_on": turned_on,
            "lever_down": lever_down,
            "lever_popped": lever_popped,
            "near_plate": near_counter,
        }

    if task == "KettleBoiling":
        grasped = obj_grasped(env, "obj")
        on_burner, burner_location, burner_distance = _kettle_on_burner(env)
        placed = on_burner and obj_far(env, "obj")
        return [grasped, placed, safe_bool(env._check_success)], {
            "kettle_grasped": grasped,
            "kettle_on_burner": on_burner,
            "kettle_burner_location": burner_location,
            "kettle_burner_distance": burner_distance,
            "kettle_placed": placed,
        }

    if task == "LoadDishwasher":
        dish0_rack = safe_bool(lambda: env.dishwasher.check_rack_contact(env, "dish0"))
        dish1_rack = safe_bool(lambda: env.dishwasher.check_rack_contact(env, "dish1"))
        return [
            obj_grasped(env, "dish0"),
            dish0_rack and obj_far(env, "dish0"),
            obj_grasped(env, "dish1"),
            dish1_rack and obj_far(env, "dish1"),
            safe_bool(env._check_success),
        ], {"dish0_rack": dish0_rack, "dish1_rack": dish1_rack}

    if task == "PackIdenticalLunches":
        m0_t0 = in_receptacle(env, "meat0", "tupperware0")
        m0_t1 = in_receptacle(env, "meat0", "tupperware1")
        m1_t0 = in_receptacle(env, "meat1", "tupperware0")
        m1_t1 = in_receptacle(env, "meat1", "tupperware1")
        v0_t0 = in_receptacle(env, "vegetable0", "tupperware0")
        v0_t1 = in_receptacle(env, "vegetable0", "tupperware1")
        v1_t0 = in_receptacle(env, "vegetable1", "tupperware0")
        v1_t1 = in_receptacle(env, "vegetable1", "tupperware1")
        meat0_placed = m0_t0 or m0_t1
        meat1_placed = m1_t0 or m1_t1
        meat0_box = 0 if m0_t0 else 1 if m0_t1 else None
        meat1_box = 0 if m1_t0 else 1 if m1_t1 else None
        meat_boxes_distinct = (
            meat0_box is not None
            and meat1_box is not None
            and meat0_box != meat1_box
        )
        veg0_with_meat = (v0_t0 and m0_t0) or (v0_t0 and m1_t0) or (v0_t1 and m0_t1) or (v0_t1 and m1_t1)
        veg1_with_meat = (v1_t0 and m0_t0) or (v1_t0 and m1_t0) or (v1_t1 and m0_t1) or (v1_t1 and m1_t1)
        near_counter = robot_near_fixture(env, getattr(env, "counter", None))
        near_fridge = robot_near_fixture(env, getattr(env, "fridge", None))
        return [
            obj_grasped(env, "meat0"),
            near_counter,
            meat0_placed and obj_far(env, "meat0"),
            near_fridge,
            obj_grasped(env, "meat1"),
            near_counter,
            meat1_placed and meat_boxes_distinct and obj_far(env, "meat1"),
            near_fridge,
            obj_grasped(env, "vegetable0"),
            near_counter,
            veg0_with_meat and obj_far(env, "vegetable0"),
            near_fridge,
            obj_grasped(env, "vegetable1"),
            near_counter,
            safe_bool(env._check_success),
        ], {
            "meat0_tupperware0": m0_t0,
            "meat0_tupperware1": m0_t1,
            "meat1_tupperware0": m1_t0,
            "meat1_tupperware1": m1_t1,
            "vegetable0_with_meat": veg0_with_meat,
            "vegetable1_with_meat": veg1_with_meat,
            "near_counter": near_counter,
            "near_fridge": near_fridge,
        }

    if task == "PreSoakPan":
        pan_in_sink = inside_fixture(env, "obj1", env.sink)
        sponge_in_sink = inside_fixture(env, "obj2", env.sink)
        water_on = bool(env.sink.get_handle_state(env=env).get("water_on", False))
        return [
            obj_grasped(env, "obj1"),
            pan_in_sink and obj_far(env, "obj1"),
            obj_grasped(env, "obj2"),
            sponge_in_sink and obj_far(env, "obj2"),
            safe_bool(env._check_success),
        ], {"pan_in_sink": pan_in_sink, "sponge_in_sink": sponge_in_sink, "water_on": water_on}

    if task == "PrepareCoffee":
        placed = safe_bool(
            lambda: env.coffee_machine.check_receptacle_placement_for_pouring(env, "obj")
        )
        turned_on = bool(getattr(env.coffee_machine, "_turned_on", False))
        return [
            obj_grasped(env, "obj"),
            placed and obj_far(env, "obj"),
            safe_bool(env._check_success),
        ], {"mug_placed": placed, "coffee_machine_turned_on": turned_on}

    if task == "RinseSinkBasin":
        state = env.sink.get_handle_state(env=env)
        washed = list(getattr(env, "washed_loc", [False, False, False]))
        return [
            bool(state.get("water_on", False)),
            safe_bool(env._check_success) or all(washed),
        ], {"water_on": bool(state.get("water_on", False)), "washed_loc": washed}

    if task == "ScrubCuttingBoard":
        contacts = int(getattr(env, "board_contact_timer", 0)) >= 5
        swept = False
        positions = getattr(env, "board_contact_positions", [])
        if positions:
            points = np.asarray(positions)
            swept = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0))) >= 0.1
        return [
            obj_grasped(env, "sponge"),
            safe_bool(env._check_success),
        ], {
            "board_contact_timer": int(getattr(env, "board_contact_timer", 0)),
            "board_sweep_valid": swept,
            "board_contact_valid": contacts,
        }

    if task == "SearingMeat":
        pan_grasped = obj_grasped(env, "pan")
        pan_on_target, pan_location, pan_distance = _stove_target_object_on_burner(env, "pan")
        near_stove = robot_near_fixture(env, getattr(env, "stove", None))
        pan_placed = pan_on_target and obj_far(env, "pan")
        meat_in_pan = in_receptacle(env, "meat", "pan", th=0.07)
        return [
            pan_grasped,
            near_stove,
            pan_placed,
            obj_grasped(env, "meat"),
            meat_in_pan and obj_far(env, "meat"),
            safe_bool(env._check_success),
        ], {
            "target_burner": getattr(env, "knob", None),
            "pan_on_burner": pan_on_target,
            "pan_burner_location": pan_location,
            "pan_burner_distance": pan_distance,
            "meat_in_pan": meat_in_pan,
        }

    if task == "SetUpCuttingStation":
        knife_on_board = in_receptacle(env, "knife", "receptacle")
        meat_on_board = in_receptacle(env, "meat", "receptacle")
        near_board = robot_near_fixture(
            env, getattr(env, "counter", None), ref_object="receptacle"
        )
        near_plate = robot_near_fixture(
            env, getattr(env, "counter", None), ref_object="meat"
        )
        return [
            obj_grasped(env, "knife"),
            near_board,
            knife_on_board and obj_far(env, "knife"),
            near_plate,
            obj_grasped(env, "meat"),
            near_board,
            safe_bool(env._check_success),
        ], {
            "knife_on_board": knife_on_board,
            "meat_on_board": meat_on_board,
            "near_cutting_board": near_board,
            "near_plate": near_plate,
        }

    if task == "StackBowlsCabinet":
        bowl1_in_cabinet = inside_fixture(env, "bowl1", env.cabinet)
        bowl2_in_cabinet = inside_fixture(env, "bowl2", env.cabinet)
        stacked_gt = in_receptacle(env, "bowl2", "bowl1")
        stacked_either = stacked_gt or in_receptacle(env, "bowl1", "bowl2")
        return [
            obj_grasped(env, "bowl1"),
            bowl1_in_cabinet and obj_far(env, "bowl1"),
            obj_grasped(env, "bowl2"),
            safe_bool(env._check_success),
        ], {
            "bowl1_in_cabinet": bowl1_in_cabinet,
            "bowl2_in_cabinet": bowl2_in_cabinet,
            "stacked_gt_direction": stacked_gt,
            "stacked_either_direction": stacked_either,
        }

    if task == "SteamInMicrowave":
        vegetable_in_bowl = in_receptacle(env, "vegetable", "bowl")
        bowl_in_microwave = inside_fixture(env, "bowl", env.microwave)
        near_microwave = robot_near_fixture(env, getattr(env, "microwave", None))
        closed = fixture_closed(env.microwave, env)
        started = bool(env.microwave.get_state().get("turned_on", False))
        return [
            obj_grasped(env, "vegetable"),
            vegetable_in_bowl and obj_far(env, "vegetable"),
            obj_grasped(env, "bowl"),
            near_microwave,
            bowl_in_microwave and obj_far(env, "bowl"),
            closed,
            safe_bool(env._check_success),
        ], {
            "vegetable_in_bowl": vegetable_in_bowl,
            "bowl_in_microwave": bowl_in_microwave,
            "near_microwave": near_microwave,
            "microwave_closed": closed,
            "microwave_started": started,
        }

    if task == "StirVegetables":
        veg1_in_pot = in_receptacle(env, "veg1", "pot")
        veg2_in_pot = in_receptacle(env, "veg2", "pot")
        return [
            obj_grasped(env, "veg1"),
            veg1_in_pot and obj_far(env, "veg1"),
            obj_grasped(env, "veg2"),
            veg2_in_pot and obj_far(env, "veg2"),
            obj_grasped(env, "spatula"),
            safe_bool(env._check_success),
        ], {
            "veg1_in_pot": veg1_in_pot,
            "veg2_in_pot": veg2_in_pot,
            "success_time": int(getattr(env, "success_time", 0)),
        }

    if task == "StoreLeftoversInBowl":
        chicken_in_bowl = in_receptacle(env, "chicken_drumstick", "bowl")
        vegetable_in_bowl = in_receptacle(env, "vegetable", "bowl")
        bowl_in_fridge = safe_bool(lambda: env.fridge.check_rack_contact(env, "bowl"))
        near_fridge = robot_near_fixture(env, getattr(env, "fridge", None))
        return [
            obj_grasped(env, "chicken_drumstick"),
            chicken_in_bowl and obj_far(env, "chicken_drumstick"),
            obj_grasped(env, "vegetable"),
            vegetable_in_bowl and obj_far(env, "vegetable"),
            obj_grasped(env, "bowl"),
            near_fridge,
            safe_bool(env._check_success),
        ], {
            "chicken_in_bowl": chicken_in_bowl,
            "vegetable_in_bowl": vegetable_in_bowl,
            "bowl_in_fridge": bowl_in_fridge,
            "near_fridge": near_fridge,
        }

    if task == "WashLettuce":
        state = env.sink.get_handle_state(env=env)
        return [
            bool(state.get("water_on", False)),
            obj_grasped(env, "lettuce"),
            safe_bool(env._check_success),
        ], {
            "water_on": bool(state.get("water_on", False)),
            "washed_time": int(getattr(env, "washed_time", 0)),
            "lettuce_under_water": safe_bool(
                lambda: env.sink.check_obj_under_water(env, "lettuce")
            ),
        }

    raise KeyError(f"No composite_seen Oracle predicate for task {task!r}")


def get_stage_catalog(
    task: str,
    manifest_path: str | Path = DEFAULT_TARGET_MANIFEST,
) -> tuple[list[GTStageSpec], str]:
    if task not in COMPOSITE_SEEN_TASKS:
        raise KeyError(f"Unsupported composite_seen task: {task}")
    return load_gt_stage_catalog(task, manifest_path)


def get_stage_labels(
    task: str,
    env: Any,
    stages: list[GTStageSpec] | None = None,
) -> tuple[dict[str, bool], dict[str, Any]]:
    """Return labels keyed by the exact stage IDs used in the Oracle plan."""
    if stages is None:
        stages, _ = get_stage_catalog(task)
    values, diagnostics = _labels_for_task(task, env)
    if len(values) != len(stages):
        raise ValueError(
            f"{task}: Oracle predicate count {len(values)} does not match "
            f"catalog stage count {len(stages)}"
        )
    labels = {
        stage.subtask_id: bool(values[index])
        for index, stage in enumerate(stages)
    }
    diagnostics = {
        **diagnostics,
        "stage_values": {
            stage.subtask_id: bool(values[index])
            for index, stage in enumerate(stages)
        },
    }
    return labels, diagnostics

