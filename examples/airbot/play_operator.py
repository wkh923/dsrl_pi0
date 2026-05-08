"""Robot operator for the AIRBOT Play robot.

Ported from VLA-RL. The airbot_data_collection package is vendored in this repo;
airbot_py / airbot_kdl / safe_arm come from Airbot/ on PYTHONPATH.

Joint actions are routed through Airbot/safe_arm.py's clamp_joint_pos (FK+IK)
so the predicted end-effector z stays >= z_min before each command leaves
Python (matches Airbot-VLA-RL/airbot/core/play_operator.py).

Requires:
  pip install pyrealsense2          (Intel RealSense SDK)
  Airbot/ on PYTHONPATH             (provides airbot_py + airbot_kdl + safe_arm)
"""
import pathlib
import sys

# Make Airbot/ importable for safe_arm + airbot_kdl + airbot_py.
# parents[2] = repo root (dsrl_pi0/).
_AIRBOT_DIR = pathlib.Path(__file__).resolve().parents[2] / "Airbot"
if str(_AIRBOT_DIR) not in sys.path:
    sys.path.insert(0, str(_AIRBOT_DIR))

from airbot_data_collection.airbot.robots.airbot_play import AIRBOTPlay
from airbot_data_collection.airbot.robots.airbot_play import AIRBOTPlayConfig
from airbot_py.arm import RobotMode
from airbot_kdl.arm_kdl import ArmKdl
from safe_arm import SafetyCounter, clamp_joint_pos, default_z_min

# Use RealSense SDK instead of V4L2 for Intel RealSense cameras
from airbot_data_collection.airbot.sensors.cameras.realsense import RealSense as CameraCls
from airbot_data_collection.common.devices.cameras.intelrealsense import IntelRealSenseCameraConfig as CameraConfig

from examples.airbot.robot_config import RobotConfig


class Robot:
    """Robot class for the AIRBOT Play robot.

    Joint actions are routed through clamp_joint_pos (FK+IK in Airbot/safe_arm.py)
    so the predicted end-effector z stays >= z_min before each command leaves
    Python.
    """

    def __init__(
        self,
        config: RobotConfig,
        z_mins: list[float] | None = None,
        eef_type: str = "G2",
    ):
        self.config = config

        # Per-arm z_min: if not overridden, look up each arm in safe_arm's
        # DEFAULT_Z_MIN_PER_ARM (the single source of truth for this machine).
        n_arms = len(self.config.robot_groups)
        if z_mins is None or len(z_mins) == 0:
            self._z_min = {n: default_z_min(n) for n in self.config.robot_groups}
        else:
            if len(z_mins) < n_arms:
                z_mins = list(z_mins) + [z_mins[-1]] * (n_arms - len(z_mins))
            elif len(z_mins) > n_arms:
                z_mins = list(z_mins)[:n_arms]
            self._z_min = dict(zip(self.config.robot_groups, z_mins))

        # FK/IK helper (single instance shared across arms, same arm_type/eef).
        self._arm_kdl = ArmKdl(arm_type="play_short", eef_type=eef_type)
        self._joint_counters = {
            name: SafetyCounter(f"SafeArm joint[{name}]")
            for name in self.config.robot_groups
        }

        self.robots = {
            name: AIRBOTPlay(AIRBOTPlayConfig(port=port))
            for name, port in zip(self.config.robot_groups, self.config.robot_ports, strict=True)
        }
        self.cameras = {
            name: CameraCls(CameraConfig(camera_index=index))
            for name, index in zip(self.config.camera_names, self.config.camera_index, strict=True)
        }
        self.keys = list(self.robots.keys()) + list(self.cameras.keys())
        self.values = list(self.robots.values()) + list(self.cameras.values())
        for key, value in zip(self.keys, self.values, strict=True):
            if not value.configure():
                raise RuntimeError(f"Failed to configure {key}.")
        # Warm-up: discard a few frames so the camera pipeline is ready
        # when the first real capture_observation() is called (which may be
        # seconds later, after model loading / user prompt).
        for name, cam in self.cameras.items():
            try:
                cam.capture_observation()
            except Exception:
                pass

    def switch_mode(self, mode):
        """Switch the mode of the robot."""
        for robot in self.robots.values():
            robot.switch_mode(mode)

    def capture_observation(self) -> dict:
        """Capture the current observation from the robot."""
        obs = {}
        for name, ins in zip(self.keys, self.values, strict=True):
            for key, value in ins.capture_observation().items():
                obs[f"{name}/{key}"] = value
        return obs

    def send_action(self, action):
        """Send the action to the robot, with FK+IK z-safety per arm."""
        for index, (group, robot) in enumerate(self.robots.items()):
            # Joint-mode commands assume SERVO_JOINT_POS.
            if robot.interface.get_control_mode() == RobotMode.SERVO_CART_POSE:
                robot.interface.switch_mode(RobotMode.SERVO_JOINT_POS)
            segment = list(action[index * 7 : (index + 1) * 7])  # [j1..j6, gripper]
            segment = clamp_joint_pos(
                segment,
                self._z_min[group],
                self._arm_kdl,
                self._joint_counters[group],
            )
            robot.send_action([float(x) for x in segment])

    def get_qpos(self, obs: dict) -> list[float]:
        """Get the joint positions of the robot."""
        qpos = []
        for group in self.config.robot_groups:
            qpos.extend(obs[f"{group}/arm/joint_state"]["data"]["position"])
            qpos.extend(obs[f"{group}/eef/joint_state"]["data"]["position"])
        return qpos

    def reset_to_pose(self, joint_positions, wait_time=3.0):
        """Reset the arm to a specified joint pose using planning mode."""
        import time
        from airbot_data_collection.basis import SystemMode
        self.switch_mode(SystemMode.RESETTING)
        time.sleep(0.1)
        for index, (_group, robot) in enumerate(self.robots.items()):
            segment = joint_positions[index * 7 : (index + 1) * 7]
            robot.send_action([float(x) for x in segment])
        time.sleep(wait_time)
        self.switch_mode(SystemMode.SAMPLING)
        time.sleep(0.1)

    def shutdown(self) -> bool:
        """Shutdown the robot."""
        for robot in self.robots.values():
            robot.shutdown()
        for camera in self.cameras.values():
            camera.shutdown()
        return True
