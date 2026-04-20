"""Configuration for the AIRBOT robot.

Copied from VLA-RL/airbot-pi0/openpi/examples/airbot/robot_config.py.
"""
from typing import Any

from pydantic import BaseModel


class RobotConfig(BaseModel):
    """Configuration for the AIRBOT robot."""

    robot_type: str = "play"
    robot_groups: list[str] | None = None
    robot_ports: list[int] = [50051, 50053]
    camera_names: list[str] = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
    camera_index: list[int | str] = [2, 4, 6]

    def model_post_init(self, __context: Any) -> None:
        default_groups = ["left", "right"]
        if self.robot_groups is None:
            self.robot_groups = default_groups[: len(self.robot_ports)]
        elif len(self.robot_groups) != len(self.robot_ports):
            raise ValueError(
                f"robot_groups length ({len(self.robot_groups)}) must match "
                f"robot_ports length ({len(self.robot_ports)})."
            )

        if len(self.camera_names) != len(self.camera_index):
            raise ValueError(
                f"camera_names length ({len(self.camera_names)}) must match "
                f"camera_index length ({len(self.camera_index)}). "
                f"camera_names={self.camera_names}, camera_index={self.camera_index}"
            )
