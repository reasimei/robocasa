"""RoboCasa LeRobot expert episode loading.

RoboCasa LeRobot 专家轨迹读取。
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ExpertEpisode:
    """All files needed to replay one expert episode."""

    dataset_root: Path
    episode_index: int
    states: np.ndarray
    model_xml: str
    ep_meta: dict[str, Any]
    actions_hdf5_order: np.ndarray
    actions_lerobot_order: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    parquet_path: Path


def find_episode_parquet(dataset_root: Path, episode_index: int) -> Path:
    """Find one episode parquet file / 查找单个 episode 的 parquet 文件。"""
    matches = sorted(
        dataset_root.glob(f"data/*/episode_{episode_index:06d}.parquet")
    )
    if not matches:
        raise FileNotFoundError(
            f"No parquet found for episode {episode_index} under {dataset_root}"
        )
    return matches[0]


def _read_parquet(path: Path) -> dict[str, list[Any]]:
    """Read only columns needed by this experiment.

    优先使用 PyArrow，避免 pandas 对部分旧 parquet 元数据的兼容性问题。
    """
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(path, columns=["action", "next.reward", "next.done"])
        columns = {
            name: table[name].to_pylist()
            for name in ("action", "next.reward", "next.done")
        }
        return columns
    except Exception as pyarrow_error:
        try:
            import pandas as pd

            frame = pd.read_parquet(path)
            return {
                "action": frame["action"].tolist(),
                "next.reward": frame["next.reward"].tolist(),
                "next.done": frame["next.done"].tolist(),
            }
        except Exception as pandas_error:
            raise RuntimeError(
                f"Unable to read parquet {path}. "
                f"PyArrow error: {pyarrow_error}; pandas error: {pandas_error}"
            ) from pandas_error


def _reorder_lerobot_action(
    actions_lerobot: np.ndarray,
    dataset_root: Path,
) -> np.ndarray:
    """Use RoboCasa's canonical LeRobot -> HDF5 action ordering."""
    from robocasa.utils.lerobot_utils import reorder_lerobot_action

    return reorder_lerobot_action(actions_lerobot, dataset_root).astype(np.float32)


def load_expert_episode(
    dataset_root: Path,
    episode_index: int,
) -> ExpertEpisode:
    """Load states, XML, metadata and actions for one episode."""
    episode_dir = dataset_root / "extras" / f"episode_{episode_index:06d}"
    states_path = episode_dir / "states.npz"
    xml_path = episode_dir / "model.xml.gz"
    meta_path = episode_dir / "ep_meta.json"
    parquet_path = find_episode_parquet(dataset_root, episode_index)
    for path in (states_path, xml_path, meta_path, parquet_path):
        if not path.exists():
            raise FileNotFoundError(path)

    states = np.load(states_path)["states"].astype(np.float64)
    with gzip.open(xml_path, "rt", encoding="utf-8") as handle:
        model_xml = handle.read()
    ep_meta = json.loads(meta_path.read_text(encoding="utf-8"))

    columns = _read_parquet(parquet_path)
    actions_lerobot = np.asarray(columns["action"], dtype=np.float32)
    if actions_lerobot.ndim != 2 or actions_lerobot.shape[1] != 12:
        raise ValueError(f"Expected [N, 12] actions, got {actions_lerobot.shape}")
    actions_hdf5 = _reorder_lerobot_action(actions_lerobot, dataset_root)
    rewards = np.asarray(columns["next.reward"], dtype=np.float64).reshape(-1)
    dones = np.asarray(columns["next.done"], dtype=bool).reshape(-1)
    if not (states.shape[0] == actions_hdf5.shape[0] == rewards.size == dones.size):
        raise ValueError(
            "Expert length mismatch: "
            f"states={states.shape[0]}, actions={actions_hdf5.shape[0]}, "
            f"rewards={rewards.size}, dones={dones.size}"
        )

    return ExpertEpisode(
        dataset_root=dataset_root,
        episode_index=episode_index,
        states=states,
        model_xml=model_xml,
        ep_meta=ep_meta,
        actions_hdf5_order=actions_hdf5,
        actions_lerobot_order=actions_lerobot,
        rewards=rewards,
        dones=dones,
        parquet_path=parquet_path,
    )

