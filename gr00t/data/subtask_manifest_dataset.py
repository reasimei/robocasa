"""Experimental manifest-backed subtask dataset.

This module is intentionally separate from ``gr00t.data.dataset``. It uses
the split manifest to expose virtual subtask episodes while leaving source
parquet and video files untouched. It is a data-access prototype, not a
drop-in replacement for the production GR00T loader yet.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import av
import numpy as np
import pandas as pd


class ManifestSubtaskDataset:
    """Read virtual subtask episodes from a standard split manifest."""

    def __init__(self, manifest_path: str | Path, camera_key: str):
        self.manifest_path = Path(manifest_path)
        payload = json.loads(self.manifest_path.read_text())
        self.source_root = Path(payload["source_root"])
        self.segments = payload["segments"]
        self.camera_key = camera_key
        self._tables: dict[int, pd.DataFrame] = {}
        self._video_cache: dict[tuple[int, int], np.ndarray] = {}
        self._video_containers: dict[tuple[int, int], Any] = {}
        self._episodes = self._load_episode_ranges()

    def _load_episode_ranges(self) -> dict[int, tuple[int, int]]:
        path = self.source_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        table = pd.read_parquet(path)
        return {
            int(row.episode_index): (int(row.dataset_from_index), int(row.dataset_to_index))
            for row in table.itertuples()
        }

    def _table(self, chunk: int) -> pd.DataFrame:
        if chunk not in self._tables:
            path = self.source_root / "data" / f"chunk-{chunk:03d}" / "file-000.parquet"
            self._tables[chunk] = pd.read_parquet(path)
        return self._tables[chunk]

    def _video(self, chunk: int):
        key = (chunk, 0)
        if key not in self._video_containers:
            path = self.source_root / "videos" / self.camera_key / f"chunk-{chunk:03d}" / "file-000.mp4"
            self._video_containers[key] = av.open(str(path))
        return self._video_containers[key]

    def __len__(self) -> int:
        return len(self.segments)

    def segment_info(self, index: int) -> dict[str, Any]:
        return self.segments[index]

    def get_state_action(self, index: int) -> dict[str, Any]:
        segment = self.segments[index]
        source_episode = int(segment["source_episode"])
        first, _ = self._episodes[source_episode]
        absolute = first + int(segment["source_frame_start"])
        chunk = absolute // int(json.loads((self.source_root / "meta" / "info.json").read_text()).get("chunks_size", 1000))
        row = self._table(chunk).iloc[absolute % self._table(chunk).shape[0]]
        return {
            "state": np.asarray(row["observation.state"], dtype=np.float32),
            "action": np.asarray(row["action"], dtype=np.float32),
            "task": segment["task"],
        }

    def read_video_frames(self, index: int, offsets: list[int]) -> list[np.ndarray]:
        """Decode frames relative to one virtual subtask segment."""
        segment = self.segments[index]
        source_episode = int(segment["source_episode"])
        first, _ = self._episodes[source_episode]
        absolute = [first + int(segment["source_frame_start"]) + x for x in offsets]
        container = self._video(0)
        wanted = set(absolute)
        output = {}
        for frame_index, frame in enumerate(container.decode(video=0)):
            if frame_index in wanted:
                output[frame_index] = frame.to_ndarray(format="rgb24")
            if len(output) == len(wanted):
                break
        return [output[x] for x in absolute]

    def close(self) -> None:
        for container in self._video_containers.values():
            container.close()
        self._video_containers.clear()
