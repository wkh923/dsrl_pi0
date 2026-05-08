#!/usr/bin/env python3
"""
Test script for AIRBOT pi0 inference with SpaceMouse intervention.

Supports both single-arm and dual-arm configurations, auto-detected from
the number of robot ports passed via --robot-config.robot_ports.

Modes:
1. AUTONOMOUS: robot(s) controlled by pi0 policy (SERVO_JOINT_POS)
2. INTERVENING: robot(s) controlled by SpaceMouse (SERVO_CART_POSE)
   - Each arm can be toggled to intervention mode independently

Safety features:
- Z-axis height protection: prevents end-effector from going below z_min
- Applies to BOTH policy output and SpaceMouse intervention
- Per-arm z_min via --z-mins (list auto-broadcasts to all arms)

Mixed mode control:
- Reset:  JOINT angles, 7 values per arm, via PLANNING mode
- Policy: JOINT angles, 7*num_arms values, via SERVO_JOINT_POS
- SpaceMouse: POSE, 8 values per arm, via SERVO_CART_POSE (direct gRPC)

Controls (per arm):
- Left button:  toggle intervention mode (Policy <-> SpaceMouse)
- Right button: toggle gripper (while intervening)
- Move SpaceMouse: control end-effector position (while intervening)

Usage — single arm (run from VLA/airbot-pi0/ with uv for openpi deps):
    cd /path/to/airbot-VLA-RL/VLA/airbot-pi0
    uv run /path/to/airbot-VLA-RL/Airbot/spacemouse/test_inference_with_spacemouse.py \
        policy-config:local-policy-config \
        --policy-config.config-path data/pick_and_place \
        --policy-config.checkpoint-dir "/path/to/checkpoint" \
        --robot-config.robot_ports 50051 \
        --robot-config.camera-index 243322074422 243522071794 \
        --reset-action -0.001618 -1.036 0.842 -1.615 0.634 1.695 0.0

Usage — dual arm:
    cd /path/to/airbot-VLA-RL/VLA/airbot-pi0
    uv run /path/to/airbot-VLA-RL/Airbot/spacemouse/test_inference_with_spacemouse.py \
        policy-config:local-policy-config \
        --policy-config.config-path data/fold_towel \
        --policy-config.checkpoint-dir "/path/to/checkpoint" \
        --robot-config.robot_ports 50051 50053 \
        --robot-config.camera-index 243222074218 243522071794 243222071389 \
        --step-rate 20 \
        --z-mins 0.002 \
        --auto-calibrate \
        --reset-action -0.001618 -1.036 0.842 -1.615 0.634 1.695 0.0 -0.001618 -1.036 0.842 -1.615 0.634 1.695 0.0

reset-action length must equal num_arms * 7 (each arm: [j1-j6, gripper]).
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from pydantic import BaseModel
import tyro

# Add paths for imports.
#   __file__ = repo/Airbot/spacemouse/test_inference_with_spacemouse.py
#   parents[1] = Airbot/      (for spacemouse package + safe_arm)
#   parents[2] = repo root
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "Airbot"))                                      # spacemouse, safe_arm
sys.path.insert(0, str(_REPO / "VLA" / "airbot-data"))                         # data_collection.*
sys.path.insert(0, str(_REPO / "VLA" / "airbot-pi0" / "airbot" / "core"))      # airbot_data_config, robot_config
# openpi: imported from VLA/airbot-pi0 — must run via `uv run` there.

# FK/IK for z-min safety in joint space
# Mock casadi to avoid ImportError from arm_kdl_ops (we only need ArmKdl from arm_kdl)
import types as _types
if "casadi" not in sys.modules:
    sys.modules["casadi"] = _types.ModuleType("casadi")
from airbot_kdl.arm_kdl import ArmKdl

# SpaceMouse imports
from spacemouse import (
    DualAirbotSpaceMouse,
    SpaceMouseState,
    create_dual_spacemouse,
)

# OpenPI imports
try:
    from data_collection.basis import System, SystemMode, ActionConfig, ObservationConfig, InterfaceType
    from data_collection.airbot.robots.airbot_play import AIRBOTPlay, AIRBOTPlayConfig
    from airbot_py.arm import RobotMode
    from data_collection.utils import init_logging
    from airbot_data_config import get_config, get_task_config
    from robot_config import RobotConfig
    from openpi.policies.policy import Policy
    from openpi.policies.policy_config import create_trained_policy

    try:
        from data_collection.airbot.sensors.cameras.v4l2 import BsonV4L2Camera as V4L2CameraCls
    except ImportError:
        from data_collection.airbot.sensors.cameras.v4l2 import V4L2Camera as V4L2CameraCls
    from data_collection.airbot.sensors.cameras.v4l2 import V4L2CameraConfig
    from data_collection.common.devices.cameras.intelrealsense import (
        IntelRealSenseCamera as RealSenseCameraCls,
        IntelRealSenseCameraConfig,
    )

    OPENPI_AVAILABLE = True
except ImportError as e:
    print(f"[Warning] OpenPI imports failed: {e}")
    print("Make sure to run from the openpi directory with 'uv run'")
    OPENPI_AVAILABLE = False


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# Z-clamp helpers — single source of truth in Airbot/safe_arm.py
from safe_arm import (
    SafetyCounter,
    clamp_cart_pose,
    clamp_joint_pos,
    default_z_min,
)


class ArmWithMixedMode:
    """Robot class for 1 or more AIRBOT Play arms with mixed mode control.

    Supports both:
    - Joint control (7*N values): [j1-j6, gripper] per arm, for Policy and Reset
    - Pose control (8*N values):  [x,y,z, qx,qy,qz,qw, gripper] per arm, for intervention

    Safety features (all routed through Airbot/safe_arm.py):
    - SpaceMouse pose intervention: clamp_cart_pose (per arm)
    - Policy joint output: clamp_joint_pos via FK+IK (per arm)
    """

    def __init__(self, config: RobotConfig, z_mins: list[float] | None = None, eef_type: str = "G2"):
        self.config = config
        # Arm names come from config.robot_groups (auto-derived from robot_ports length).
        self.arm_names = list(config.robot_groups)
        self.num_arms = len(self.arm_names)

        # If not overridden, look up each arm's z_min from safe_arm's per-arm
        # defaults (DEFAULT_Z_MIN_PER_ARM); single source of truth for this machine.
        if z_mins is None or len(z_mins) == 0:
            self.z_min = {name: default_z_min(name) for name in self.arm_names}
        else:
            if len(z_mins) < self.num_arms:
                z_mins = list(z_mins) + [z_mins[-1]] * (self.num_arms - len(z_mins))
            elif len(z_mins) > self.num_arms:
                z_mins = list(z_mins)[: self.num_arms]
            self.z_min = {name: z_mins[i] for i, name in enumerate(self.arm_names)}
        self._arm_kdl = ArmKdl(arm_type="play_short", eef_type=eef_type)
        self._cart_counters = {name: SafetyCounter(f"SafeArm cart[{name}]") for name in self.arm_names}
        self._joint_counters = {name: SafetyCounter(f"SafeArm joint[{name}]") for name in self.arm_names}

        # Create robot config with default (joint) action mode
        # but also enable pose observation for SpaceMouse intervention
        self.robots = {}
        for name, port in zip(self.config.robot_groups, self.config.robot_ports, strict=True):
            robot_config = AIRBOTPlayConfig(
                port=port,
                # Default action uses joint angles (SERVO_JOINT)
                action=[
                    ActionConfig(),  # arm uses default (joint)
                    ActionConfig(),  # eef uses default
                ],
                # Enable both joint and pose observation
                observation=[
                    ObservationConfig(interfaces={InterfaceType.JOINT_POSITION, InterfaceType.POSE}),
                    ObservationConfig(),  # eef uses default
                ],
            )
            self.robots[name] = AIRBOTPlay(robot_config)

        self.cameras = {}
        for name, index in zip(self.config.camera_names, self.config.camera_index, strict=True):
            index_str = str(index)
            if len(index_str) > 5:
                self.cameras[name] = RealSenseCameraCls(IntelRealSenseCameraConfig(camera_index=index_str))
            else:
                self.cameras[name] = V4L2CameraCls(V4L2CameraConfig(camera_index=index))
        self.keys = list(self.robots.keys()) + list(self.cameras.keys())
        self.values = list(self.robots.values()) + list(self.cameras.values())
        for key, value in zip(self.keys, self.values, strict=True):
            if not value.configure():
                raise RuntimeError(f"Failed to configure {key}.")

    def switch_mode(self, mode):
        """Switch the mode of all robots."""
        for robot in self.robots.values():
            robot.switch_mode(mode)

    def capture_observation(self) -> dict:
        """Capture the current observation from all robots and cameras."""
        obs = {}
        for name, ins in zip(self.keys, self.values, strict=True):
            for key, value in ins.capture_observation().items():
                obs[f"{name}/{key}"] = value
        return obs

    def send_joint_action(self, action, check_safety: bool = True):
        """Send joint action to all robots (for Policy and Reset).

        Action format: [left_j1-j6, left_gripper, right_j1-j6, right_gripper] (14 values)
        """
        for index, (group, robot) in enumerate(self.robots.items()):
            current_mode = robot.interface.get_control_mode()
            if current_mode == RobotMode.SERVO_CART_POSE:
                robot.interface.switch_mode(RobotMode.SERVO_JOINT_POS)

            arm_input = list(action[index * 7 : index * 7 + 7])  # [j1..j6, gripper]
            if check_safety:
                arm_action = clamp_joint_pos(
                    arm_input,
                    self.z_min[group],
                    self._arm_kdl,
                    self._joint_counters[group],
                )
            else:
                arm_action = arm_input

            robot.send_action(arm_action)

            # Post-execution verification: if clamp_joint_pos modified the joints,
            # read real end-effector z to confirm the arm actually reached safe height.
            corrected = arm_action != arm_input
            counter = self._joint_counters[group]
            if corrected and (counter.count <= 10 or counter.count % 100 == 0):
                try:
                    end_pose = robot.interface.get_end_pose()
                    if end_pose:
                        real_z = end_pose[0][2]
                        z_min = self.z_min[group]
                        logger.warning(
                            f"[Verify {group}] real_z={real_z:.4f}, target={z_min:.4f}, "
                            f"diff={real_z - z_min:.4f}"
                        )
                except Exception:
                    pass

    def send_pose_action_single(self, arm_name: str, pose_action: list):
        """Send pose action to a single robot arm (for SpaceMouse intervention).

        Args:
            arm_name: one of self.arm_names (e.g. "follow" / "left" / "right")
            pose_action: [x, y, z, qx, qy, qz, qw, gripper] (8 values)

        Safety: Z coordinate is clamped to z_min before sending.
        """
        if arm_name not in self.robots:
            logger.warning(f"Unknown arm: {arm_name}")
            return

        robot = self.robots[arm_name]

        try:
            # Switch to SERVO_CART_POSE mode if not already
            current_mode = robot.interface.get_control_mode()
            if current_mode != RobotMode.SERVO_CART_POSE:
                robot.interface.switch_mode(RobotMode.SERVO_CART_POSE)
            # Apply safety clamping (Cartesian z >= z_min)
            pos = [float(x) for x in pose_action[:3]]
            quat = [float(x) for x in pose_action[3:7]]
            pos, quat = clamp_cart_pose(
                [pos, quat], self.z_min[arm_name], self._cart_counters[arm_name]
            )
            robot.interface.servo_cart_pose([pos, quat])
            robot.interface.servo_eef_pos([float(pose_action[7])])  # gripper
        except Exception as e:
            logger.warning(f"Failed to send pose action to {arm_name}: {e}")

    def send_pose_action(self, action):
        """Send pose action to all robots (for SpaceMouse intervention).

        Action format: [left_pose(8), right_pose(8)] (16 values total)

        Safety: Z coordinates are clamped to z_min before sending.
        """
        for index, (group, robot) in enumerate(self.robots.items()):
            # Pose action: 7 (pose) + 1 (gripper) = 8 per robot
            pose_data = list(action[index * 8 : (index + 1) * 8])

            try:
                # Switch to SERVO_CART_POSE mode if not already
                current_mode = robot.interface.get_control_mode()
                if current_mode != RobotMode.SERVO_CART_POSE:
                    robot.interface.switch_mode(RobotMode.SERVO_CART_POSE)
                # Apply safety clamping (Cartesian z >= z_min)
                pos = [float(x) for x in pose_data[:3]]
                quat = [float(x) for x in pose_data[3:7]]
                pos, quat = clamp_cart_pose(
                    [pos, quat], self.z_min[group], self._cart_counters[group]
                )
                robot.interface.servo_cart_pose([pos, quat])
                robot.interface.servo_eef_pos([float(pose_data[7])])  # gripper
            except Exception as e:
                logger.warning(f"Failed to send pose action to {group}: {e}")

    def send_action(self, action):
        """Send action to all robots (default: joint mode).

        For backward compatibility. Use send_joint_action() or send_pose_action() explicitly.
        """
        self.send_joint_action(action)

    def get_qpos(self, obs: dict) -> list[float]:
        """Get the joint state + gripper of all robots.

        Returns: [left_j1-j6, left_gripper, right_j1-j6, right_gripper] (14 values)
        """
        qpos = []
        for group in self.config.robot_groups:
            # Get joint positions
            joint_state = obs[f"{group}/arm/joint_state"]["data"]
            qpos.extend(joint_state["position"])
            # Get gripper
            qpos.extend(obs[f"{group}/eef/joint_state"]["data"]["position"])
        return qpos

    def get_pose(self, obs: dict) -> list[float]:
        """Get the pose + gripper state of all robots.

        Returns: [left_pose(8), right_pose(8)] (16 values total)
        """
        pose_data = []
        for group in self.config.robot_groups:
            # Get pose
            pose = obs[f"{group}/arm/pose"]["data"]
            pose_data.extend(pose["position"])
            pose_data.extend(pose["orientation"])
            # Get gripper
            pose_data.extend(obs[f"{group}/eef/joint_state"]["data"]["position"])
        return pose_data

    def get_pose_single(self, obs: dict, arm_name: str) -> Optional[tuple]:
        """Get the pose of a single arm.

        Args:
            obs: Observation dict
            arm_name: one of self.arm_names (e.g. "follow" / "left" / "right")

        Returns:
            (position, orientation, gripper) or (None, None, None) if not available
        """
        pose_key = f"{arm_name}/arm/pose"
        gripper_key = f"{arm_name}/eef/joint_state"

        if pose_key in obs:
            pose_data = obs[pose_key]["data"]
            position = pose_data["position"]
            orientation = pose_data["orientation"]
        else:
            logger.warning(f"{pose_key} not in observation. Make sure pose observation is enabled.")
            return None, None, None

        gripper = 0.0
        if gripper_key in obs:
            gripper = obs[gripper_key]["data"]["position"][0]

        return position, orientation, gripper

    def shutdown(self) -> bool:
        """Shutdown all robots and cameras."""
        for robot in self.robots.values():
            robot.shutdown()
        for camera in self.cameras.values():
            camera.shutdown()
        return True


class LocalPolicyConfig(BaseModel):
    config_path: str
    checkpoint_dir: str
    seed: int = -1


class RemotePolicyConfig(BaseModel):
    host: str = "localhost"
    port: int = 8000


class InferConfig(BaseModel):
    policy_config: LocalPolicyConfig | RemotePolicyConfig
    max_steps: int = 250000
    step_rate: int = 20
    # Per-arm step length: [j1-j6, gripper]. Auto-tiled to num_arms at runtime.
    step_length: list[float] = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.05]
    # Reset action: joint targets. Length must equal num_arms * 7.
    # Default shows single-arm; for dual arm pass 14 values on the CLI.
    reset_action: list[float] = [
        0.0, 0.131, 0.0, 0.0, -0.331, 0.0,  # j1-j6 (radians)
        0.04,                                # gripper (0=open, ~0.07=closed)
    ]
    interpolate: bool = False
    chunk_size_execute: int = 25
    debug: bool = False
    prompt: str = ""
    robot_config: RobotConfig

    # SpaceMouse settings
    spacemouse_pos_scale: float = 0.3     # Position delta scale (meters per unit)
    spacemouse_rot_scale: float = 0.3     # Rotation delta scale (radians per unit)
    auto_calibrate: bool = False          # Auto-calibrate SpaceMouse assignment

    # Safety settings — one value per arm. Auto-broadcast to num_arms at runtime.
    z_mins: list[float] = [0.001]         # Minimum z height per arm (meters, base frame)
    eef_type: str = "G2"                  # End effector type for FK/IK (G2, E2B, none, etc.)


class AutoConfig(BaseModel):
    chunk_size_predict: int = 0
    state_dim: int = -1
    camera_names: list[str] = []
    observation: dict = {"qpos": None, "images": {}}


auto_config = AutoConfig()


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> list:
    """Convert Euler angles (roll, pitch, yaw) to quaternion (x, y, z, w)."""
    cr, cp, cy = np.cos(roll/2), np.cos(pitch/2), np.cos(yaw/2)
    sr, sp, sy = np.sin(roll/2), np.sin(pitch/2), np.sin(yaw/2)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return [qx, qy, qz, qw]


def quaternion_to_euler(qx: float, qy: float, qz: float, qw: float) -> tuple:
    """Convert quaternion (x, y, z, w) to Euler angles (roll, pitch, yaw)."""
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def quaternion_multiply(q1: list, q2: list) -> list:
    """Multiply two quaternions q1 * q2. Format: [x, y, z, w]."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return [
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ]


def update_observation(camera_names: list[str], operator: System):
    """Update the observation from the robot."""
    obs = operator.capture_observation()
    qpos = operator.get_qpos(obs)
    image_dict = {}
    for camera_name in camera_names:
        raw = obs[f"{camera_name}/color/image_raw"]
        image_dict[f"observation/{camera_name}"] = raw["data"] if isinstance(raw, dict) else raw
    auto_config.observation = {"qpos": np.array(qpos), "images": image_dict}


def inference_once(policy: Policy, prompt: str) -> np.ndarray:
    """Perform a single inference step using the trained policy."""
    obs = {"observation/state": auto_config.observation["qpos"], "prompt": prompt} | auto_config.observation["images"]
    action_chunk = policy.infer(obs)["actions"] if policy is not None else np.zeros([64, 14], dtype=np.float32)
    auto_config.chunk_size_predict = action_chunk.shape[0]
    auto_config.state_dim = action_chunk.shape[1]
    return action_chunk


def inference_with_spacemouse(config: InferConfig, operator: ArmWithMixedMode):
    """Main inference loop with SpaceMouse intervention support (1 or N arms).

    Mixed mode control:
    - Policy/Reset: Joint angles (7 * num_arms values) via SERVO_JOINT_POS
    - SpaceMouse: Pose (8 values per arm) via direct gRPC (SERVO_CART_POSE)

    Each arm can be independently switched to intervention mode.
    Safety protection (z_min) applies to both policy and intervention for every arm.
    """

    auto_config.camera_names = operator.config.camera_names
    policy_config = config.policy_config
    arm_names = list(operator.arm_names)
    num_arms = len(arm_names)

    # Initialize policy
    if isinstance(policy_config, LocalPolicyConfig):
        if policy_config.seed >= 0:
            torch.manual_seed(policy_config.seed)
            np.random.seed(policy_config.seed)
        task_config = get_task_config(policy_config.config_path)
        policy = create_trained_policy(get_config(task_config), policy_config.checkpoint_dir)
        if not config.prompt:
            config.prompt = task_config.task_name
    else:
        from openpi_client import websocket_client_policy
        policy = websocket_client_policy.WebsocketClientPolicy(
            host=policy_config.host, port=policy_config.port
        )
        assert config.prompt, "Prompt must be provided for remote policy inference."

    # On takeover the driver calls this to align its internal gripper
    # toggle to the robot's current width — avoids spurious open/close on
    # the first INTERVENING step.
    # Initialize SpaceMouse (supports num_arms=1 or more)
    logger.info(f"Initializing SpaceMouse(num_arms={num_arms})...")
    spacemouse = create_dual_spacemouse(
        num_arms=num_arms,
        use_mock=False,
        auto_calibrate=config.auto_calibrate,
        pos_scale=config.spacemouse_pos_scale,
        rot_scale=config.spacemouse_rot_scale,
    )

    # Push current real gripper state every step so the driver has a fresh
    # snapshot at takeover. Compare distance to the same gripper_open /
    # gripper_closed constants that the intervention handler uses below,
    # so the takeover snapshot reproduces the value the downstream sends.
    def _push_gripper_states():
        for arm_idx, robot_inst in enumerate(operator.robots.values()):
            try:
                pos = robot_inst.interface.get_eef_pos()
                if pos:
                    closed = abs(pos[0] - gripper_closed) < abs(pos[0] - gripper_open)
                    spacemouse.update_gripper_state(arm_idx, closed)
            except Exception:
                pass
    logger.info(f"SpaceMouse type: {type(spacemouse).__name__}")

    z_min_str = ", ".join(f"{n}={operator.z_min[n]:.4f}m" for n in arm_names)

    print("\n" + "=" * 70)
    print(f"SpaceMouse Intervention Test — {num_arms} arm(s) (Mixed Mode)")
    print("=" * 70)
    print("Controls (per arm):")
    print("  - Left button:  Toggle intervention mode")
    print("                  AUTONOMOUS (Policy) <-> INTERVENING (SpaceMouse)")
    print("  - Right button: Toggle gripper (while intervening)")
    print("  - Move SpaceMouse: Control robot end-effector position")
    print("                     (only effective when intervening)")
    print("")
    print(f"Policy/Reset: Joint angles [j1-j6, gripper] x {num_arms} arms ({num_arms*7} values)")
    print("SpaceMouse:   Pose [x,y,z, qx,qy,qz,qw, gripper] per arm (8 values)")
    print(f"Safety:       z_min {z_min_str}")
    print("=" * 70 + "\n")

    # Statistics
    total_steps = 0
    intervention_steps = {name: 0 for name in arm_names}
    autonomous_steps = {name: 0 for name in arm_names}

    # Gripper settings
    gripper_open = 0.0
    gripper_closed = 0.07

    try:
        while True:
            # Reset to initial position using joint angles
            operator.switch_mode(SystemMode.RESETTING)
            operator.send_joint_action(config.reset_action)  # 14 values
            spacemouse.reset()  # Reset SpaceMouse to AUTONOMOUS mode

            user_input = input("\nPress 'Enter' to start episode, 'q' to quit: ")
            if user_input.lower() in {'q', 'z'}:
                logger.info("Quitting...")
                break

            operator.switch_mode(SystemMode.SAMPLING)

            with torch.inference_mode():
                pre_action = np.array(config.reset_action)
                update_observation(auto_config.camera_names, operator)
                t = 0
                action_chunk = None
                just_exited_intervention = [False] * num_arms  # Per arm

                while t < config.max_steps:
                    # Keep driver's snapshot of real gripper state fresh.
                    _push_gripper_states()

                    # Get SpaceMouse action for all arms
                    # action format: [arm0_dx,dy,dz,droll,dpitch,dyaw,gripper, arm1_...]
                    sm_action, arm_states, state_changed = spacemouse.get_action()

                    # Check intervention state for each arm
                    is_intervening = [
                        arm_states[i] == SpaceMouseState.INTERVENING
                        for i in range(num_arms)
                    ]
                    any_intervening = any(is_intervening)

                    # Handle mode switching
                    if state_changed:
                        print(f"\n{'='*50}")
                        for i, name in enumerate(arm_names):
                            state_str = "INTERVENING" if is_intervening[i] else "AUTONOMOUS"
                            print(f"  {name.upper()} arm: {state_str}")

                            if not is_intervening[i] and arm_states[i] == SpaceMouseState.AUTONOMOUS:
                                # Just exited intervention for this arm
                                just_exited_intervention[i] = True
                        print(f"{'='*50}\n")

                        # If all arms are now autonomous, reset action chunk
                        if not any_intervening:
                            action_chunk = None
                            # Update pre_action to current joint state for smooth transition
                            obs = operator.capture_observation()
                            current_qpos = operator.get_qpos(obs)
                            pre_action = np.array(current_qpos)
                            logger.info("All arms autonomous, waiting for stabilization...")
                            time.sleep(0.5)
                            update_observation(auto_config.camera_names, operator)
                            logger.info("Resuming policy control")

                    # Execute action based on mode
                    obs = operator.capture_observation()

                    # Process each arm
                    for i, name in enumerate(arm_names):
                        if is_intervening[i]:
                            # SpaceMouse intervention for this arm
                            intervention_steps[name] += 1

                            pos, ori, gripper = operator.get_pose_single(obs, name)

                            if pos is not None and ori is not None:
                                # sm_action: [dx, dy, dz, droll, dpitch, dyaw, gripper_toggle] per arm
                                offset = i * 7
                                arm_sm_action = sm_action[offset:offset + 7]

                                # Apply position delta (z safety is enforced downstream
                                # by send_pose_action_single via clamp_cart_pose)
                                new_pos = [
                                    pos[0] + arm_sm_action[0],
                                    pos[1] + arm_sm_action[1],
                                    pos[2] + arm_sm_action[2],
                                ]

                                # Apply rotation delta directly in quaternion space (avoids gimbal lock)
                                delta_quat = euler_to_quaternion(arm_sm_action[3], arm_sm_action[4], arm_sm_action[5])
                                new_quat = quaternion_multiply(delta_quat, [ori[0], ori[1], ori[2], ori[3]])
                                # Normalize to prevent drift
                                norm = np.sqrt(sum(x*x for x in new_quat))
                                new_quat = [x / norm for x in new_quat]

                                # Gripper
                                gripper_width = gripper_closed if arm_sm_action[6] > 0.5 else gripper_open

                                # Construct pose action
                                pose_action = new_pos + new_quat + [gripper_width]

                                # Send via pose control (already has safety clamping inside)
                                operator.send_pose_action_single(name, pose_action)

                                if t % 20 == 0:
                                    logger.info(
                                        f"[INTERVENTION {name.upper()}] Step {t}: "
                                        f"pos=[{new_pos[0]:.3f}, {new_pos[1]:.3f}, {new_pos[2]:.3f}] "
                                        f"gripper={gripper_width:.3f}"
                                    )
                            else:
                                logger.warning(f"Could not get current pose for {name} arm intervention")
                        else:
                            # Autonomous mode for this arm
                            autonomous_steps[name] += 1

                    # If no arm is intervening, use policy action for all
                    if not any_intervening:
                        action_index = t % config.chunk_size_execute

                        # Refresh action chunk when needed
                        if action_index == 0 or action_chunk is None:
                            update_observation(auto_config.camera_names, operator)
                            start_time = time.monotonic()
                            logger.info("Policy inference...")
                            action_chunk = inference_once(policy, config.prompt).copy()
                            logger.info(f"Inference time: {time.monotonic() - start_time:.3f}s")

                        action = action_chunk[action_index]

                        # Pad pre_action to match action dimension if needed
                        if len(action) > len(pre_action):
                            pre_action = np.pad(pre_action, (0, len(action) - len(pre_action)), mode='constant')

                        # Force interpolation after exiting intervention for smooth transition
                        use_interpolate = config.interpolate or any(just_exited_intervention)
                        if use_interpolate:
                            step_length = np.array(config.step_length)
                            if len(action) > len(step_length):
                                step_length = np.pad(step_length, (0, len(action) - len(step_length)), mode='edge')
                            diff = np.abs(action - pre_action)
                            steps = int(np.ceil(np.max(diff / step_length)))
                            if steps > 1:
                                interp_actions = np.linspace(pre_action, action, steps + 1)[1:]
                                if any(just_exited_intervention):
                                    logger.info(f"Smooth transition: {steps} interpolation steps")
                            else:
                                interp_actions = action[np.newaxis, :]
                            just_exited_intervention = [False] * num_arms
                        else:
                            interp_actions = action[np.newaxis, :]

                        for act in interp_actions:
                            operator.send_joint_action(act)  # 14 values
                            time.sleep(1.0 / config.step_rate)

                        pre_action = action.copy()

                        if t % 50 == 0:
                            logger.info(f"[AUTONOMOUS] Step {t}: Policy action index = {action_index}")
                    else:
                        # At least one arm is intervening, wait for next step
                        time.sleep(1.0 / config.step_rate)

                    t += 1
                    total_steps += 1

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    finally:
        # Print statistics
        print("\n" + "=" * 70)
        print("Session Statistics")
        print("=" * 70)
        print(f"Total steps:           {total_steps}")
        for name in arm_names:
            auto_pct = 100 * autonomous_steps[name] / max(1, total_steps)
            int_pct = 100 * intervention_steps[name] / max(1, total_steps)
            print(f"{name.upper()} arm:")
            print(f"  Autonomous steps:    {autonomous_steps[name]} ({auto_pct:.1f}%)")
            print(f"  Intervention steps:  {intervention_steps[name]} ({int_pct:.1f}%)")
        print("=" * 70)

        spacemouse.close()
        operator.shutdown()


def main():
    if not OPENPI_AVAILABLE:
        print("\nError: OpenPI modules not available.")
        print("Please run this script from the openpi directory:")
        print("  cd /path/to/openpi")
        print("  uv run /path/to/airbot-VLA-RL/Airbot/spacemouse/test_inference_with_spacemouse.py ...")
        sys.exit(1)

    config = tyro.cli(InferConfig, config=[tyro.conf.ConsolidateSubcommandArgs])

    # Validate reset_action length matches num_arms * 7
    num_arms = len(config.robot_config.robot_ports)
    expected_len = num_arms * 7
    if len(config.reset_action) != expected_len:
        print(f"\nError: reset_action should have {expected_len} values "
              f"({num_arms} arms x 7 values each)")
        print(f"Got {len(config.reset_action)} values: {config.reset_action}")
        sys.exit(1)

    robot_config = config.robot_config
    if robot_config.robot_type != "play":
        raise ValueError(f"Unsupported robot type: {robot_config.robot_type}. Only 'play' is supported.")

    robot = ArmWithMixedMode(
        robot_config,
        z_mins=config.z_mins,
        eef_type=config.eef_type,
    )

    inference_with_spacemouse(config, robot)


if __name__ == "__main__":
    main()
