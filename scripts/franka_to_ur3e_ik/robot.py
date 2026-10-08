"""UR3e robot model used by the Franka-to-UR3e experiment.

本文件只定义本实验使用的 UR3e 模型，不修改 RoboCasa / RoboSuite 源码。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel


_XML_PATH = Path(__file__).with_name("ur3e_official.xml")


class UR3eFrankaReplay(ManipulatorModel):
    """Fixed-base UR3e with a Robotiq 2F-85 gripper.

    固定底座 UR3e，末端安装 Robotiq 2F-85 夹爪。
    """

    arms = ["right"]

    def __init__(self, idn: int = 0):
        super().__init__(str(_XML_PATH), idn=idn)

    @property
    def default_base(self):
        return "NullMount"

    @property
    def default_gripper(self):
        return {"right": "Robotiq85Gripper"}

    @property
    def default_controller_config(self):
        return {"right": "default_ur5e"}

    @property
    def init_qpos(self):
        # This is only the IK seed before the initial EEF alignment.
        # 这里只作为初始 IK 种子，随后会对齐到 Franka 的起始末端位姿。
        return np.array(
            [-0.35, -1.35, 1.75, -1.95, -1.57, -1.55],
            dtype=np.float64,
        )

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

