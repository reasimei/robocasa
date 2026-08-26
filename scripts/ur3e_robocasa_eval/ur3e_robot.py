"""Minimal UR3e MuJoCo model for fixed-base RoboCasa experiments.

The kinematic dimensions follow the UR3e description shipped with Isaac Sim.
The link visuals are intentionally lightweight primitives so this experiment
does not depend on another robot asset package.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel


_XML_PATH = Path(__file__).with_name("ur3e.xml")


class UR3e(ManipulatorModel):
    arms = ["right"]

    def __init__(self, idn=0):
        super().__init__(str(_XML_PATH), idn=idn)

    @property
    def default_base(self):
        # The kitchen placement code places this model in front of the sink.
        # A null mount makes it a suspended fixed-base arm at the configured z.
        return "NullMount"

    @property
    def default_gripper(self):
        return {"right": "Robotiq85Gripper"}

    @property
    def default_controller_config(self):
        return {"right": "default_ur5e"}

    @property
    def init_qpos(self):
        # A standard elbow-down UR3e posture, adjusted for the shorter links.
        return np.array([-0.35, -1.35, 1.75, -1.95, -1.57, -1.55], dtype=np.float64)

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.4, 0.0, 0),
            "empty": (-0.4, 0.0, 0),
            "table": lambda table_length: (-0.4 - table_length / 2, 0, 0),
        }

    @property
    def top_offset(self):
        return np.array((0.0, 0.0, 0.65), dtype=np.float64)

    @property
    def _horizontal_radius(self):
        return 0.45

    @property
    def arm_type(self):
        return "single"
