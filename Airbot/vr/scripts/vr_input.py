#!/usr/bin/env python3
"""
Quest3 VR Controller input for AIRBOT dual-arm teleoperation.

Subscribes to Quest3 ROS2 topics and produces the same output interface
as DualAirbotSpaceMouse.get_action(), making it a drop-in replacement.

ROS2 Topics:
    /tf (tf2_msgs/TFMessage) — hand poses (child_frame_id: hand_left, hand_right)
    /quest/joystick (sensor_msgs/Joy) — 8 axes + 17 buttons

Control approach (hybrid delta mode):
    - Press grip button once to engage: VR controller movement → robot deltas
    - Press grip button again to disengage (clutch): robot holds position
    - Trigger toggles gripper open/close

Usage:
    from vr_input import DualVRInput

    vr = DualVRInput(pos_scale=1.0, rot_scale=1.0)
    action, states, changed = vr.get_action()
    # action: 16D [left_8D, right_8D], each 8D = [dx,dy,dz,qx,qy,qz,qw,gripper]
"""

import threading
import time
from enum import Enum
from typing import Tuple, Optional, List, Dict
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import Joy


class VRState(Enum):
    """State of VR controller intervention (matches SpaceMouseState interface)."""
    AUTONOMOUS = 0   # Grip released — robot holds position
    INTERVENING = 1  # Grip squeezed — VR controls robot


# Axis indices in /quest/joystick axes[] array (8 axes).
class QuestAxes:
    LEFT_STICK_X = 0     # Left stick horizontal
    LEFT_STICK_Y = 1     # Left stick vertical
    RIGHT_STICK_X = 2    # Right stick horizontal
    RIGHT_STICK_Y = 3    # Right stick vertical
    LEFT_TRIGGER = 4     # Left trigger (continuous 0-1)
    RIGHT_TRIGGER = 5    # Right trigger (continuous 0-1)
    LEFT_GRIP = 6        # Left grip (continuous 0-1)
    RIGHT_GRIP = 7       # Right grip (continuous 0-1)


# Button indices in /quest/joystick buttons[] array (17 buttons).
class QuestButtons:
    X_BUTTON = 0              # X button (left controller)
    A_BUTTON = 1              # A button (right controller)
    B_BUTTON = 2              # B button (right controller)
    Y_BUTTON = 3              # Y button (left controller)
    # 4, 5, 6, 7 — unknown/unused
    LEFT_TRIGGER_BUTTON = 8   # Left trigger (bool)
    RIGHT_TRIGGER_BUTTON = 9  # Right trigger (bool)
    LEFT_GRIP_BUTTON = 10     # Left grip (bool)
    RIGHT_GRIP_BUTTON = 11    # Right grip (bool)
    LEFT_STICK_CLICK = 12     # Left stick click
    RIGHT_STICK_CLICK = 13    # Right stick click


def quat_multiply(q1: List[float], q2: List[float]) -> List[float]:
    """Multiply two quaternions [qx, qy, qz, qw]."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return [
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ]


def quat_inverse(q: List[float]) -> List[float]:
    """Inverse of a unit quaternion [qx, qy, qz, qw] = conjugate."""
    return [-q[0], -q[1], -q[2], q[3]]


def quat_normalize(q: List[float]) -> List[float]:
    """Normalize a quaternion to unit length."""
    norm = np.sqrt(sum(x*x for x in q))
    if norm < 1e-10:
        return [0.0, 0.0, 0.0, 1.0]
    return [x / norm for x in q]


def quat_slerp(q1: List[float], q2: List[float], t: float) -> List[float]:
    """Spherical linear interpolation between quaternions [qx, qy, qz, qw]."""
    dot = sum(a * b for a, b in zip(q1, q2))
    # Ensure shortest path
    if dot < 0:
        q2 = [-x for x in q2]
        dot = -dot
    # Very close — fall back to normalized lerp
    if dot > 0.9995:
        result = [a + t * (b - a) for a, b in zip(q1, q2)]
    else:
        theta = np.arccos(np.clip(dot, -1.0, 1.0))
        sin_theta = np.sin(theta)
        s1 = np.sin((1 - t) * theta) / sin_theta
        s2 = np.sin(t * theta) / sin_theta
        result = [s1 * a + s2 * b for a, b in zip(q1, q2)]
    return quat_normalize(result)


class DualVRInput:
    """
    Quest3 VR controller input for dual-arm robot control.

    Subscribes to Quest3 ROS2 topics and provides the same interface
    as DualAirbotSpaceMouse.get_action().

    Hybrid delta mode:
        - Squeeze grip → record anchor pose, compute frame-to-frame deltas
        - Release grip → zero output, arm holds position
        - Trigger → toggle gripper

    Args:
        pos_scale: Position delta scaling factor (meters)
        rot_scale: Rotation delta scaling factor (radians)
        grip_threshold: Grip axis value to consider "engaged" (0.0-1.0)
        tf_topic: ROS2 topic for TF transforms
        joy_topic: ROS2 topic for joystick data
        left_grip_axis: Index of left grip in joy axes array
        right_grip_axis: Index of right grip in joy axes array
        left_trigger_button: Index of left trigger in joy buttons array
        right_trigger_button: Index of right trigger in joy buttons array
    """

    ARM_NAMES = ["left", "right"]

    # TF child frame IDs for each hand
    TF_FRAMES = {"left": "hand_left", "right": "hand_right"}

    def __init__(
        self,
        pos_scale: float = 1.0,
        rot_scale: float = 1.0,
        grip_threshold: float = 0.5,
        tf_topic: str = "/tf",
        joy_topic: str = "/quest/joystick",
        left_grip_axis: int = QuestAxes.LEFT_GRIP,       # axes[6]
        right_grip_axis: int = QuestAxes.RIGHT_GRIP,      # axes[7]
        left_trigger_button: int = QuestButtons.LEFT_TRIGGER_BUTTON,   # buttons[8]
        right_trigger_button: int = QuestButtons.RIGHT_TRIGGER_BUTTON, # buttons[9]
        settle_duration: float = 0.5,
        init_ros: bool = True,
    ):
        self.pos_scale = pos_scale
        self.rot_scale = rot_scale
        self.grip_threshold = grip_threshold
        self._settle_duration = settle_duration
        self.num_arms = 2

        # Axis/button index config (adjustable for different mappings)
        self._grip_axes = [left_grip_axis, right_grip_axis]
        self._trigger_buttons = [left_trigger_button, right_trigger_button]

        # Per-arm state
        self._arm_states: List[VRState] = [VRState.AUTONOMOUS, VRState.AUTONOMOUS]
        self._gripper_states: List[float] = [0.0, 0.0]  # 0=open, 1=closed

        # Edge detection for grip toggle and trigger buttons
        self._prev_grip: List[bool] = [False, False]
        self._prev_trigger: List[bool] = [False, False]

        # Settle period: timestamp when each arm was engaged (None = not engaged)
        self._engage_time: List[Optional[float]] = [None, None]

        # Pose tracking for frame-to-frame deltas
        # _prev_pose[arm] = (position, quaternion) from the previous frame while gripped
        self._prev_pose: List[Optional[Tuple[np.ndarray, np.ndarray]]] = [None, None]

        # Latest TF data (updated by ROS2 callback)
        self._lock = threading.Lock()
        self._hand_poses: Dict[str, Optional[Tuple[np.ndarray, np.ndarray]]] = {
            "left": None,
            "right": None,
        }
        # Timestamped pose buffer per side: list of (recv_time, pos, quat).
        # Used by _interp_pose to absorb bursty arrivals from Quest.
        # Lag (LERP_LAG) > max observed inter-arrival (~54ms) avoids extrapolation.
        from collections import deque
        self._pose_buf: Dict[str, "deque"] = {
            "left": deque(maxlen=64),
            "right": deque(maxlen=64),
        }
        self.LERP_LAG_S: float = 0.06   # 60ms lag — flat-out absorbs the 54ms max gap
        # Latest joystick data
        self._joy_axes: List[float] = [0.0] * 8
        self._joy_buttons: List[int] = [0] * 17

        # ROS2 setup
        self._init_ros = init_ros
        if init_ros and not rclpy.ok():
            rclpy.init()

        self._node = Node("dual_vr_input")
        self._node.get_logger().info("DualVRInput node created")

        # Subscribe to TF
        self._tf_sub = self._node.create_subscription(
            TFMessage,
            tf_topic,
            self._tf_callback,
            10,
        )

        # Subscribe to joystick
        self._joy_sub = self._node.create_subscription(
            Joy,
            joy_topic,
            self._joy_callback,
            10,
        )

        # Spin ROS2 in background thread
        self._executor = MultiThreadedExecutor(num_threads=2)
        self._executor.add_node(self._node)
        self._spinning = True
        self._spin_thread = threading.Thread(target=self._spin_ros, daemon=True)
        self._spin_thread.start()

        self._node.get_logger().info(
            f"DualVRInput initialized: tf={tf_topic}, joy={joy_topic}, "
            f"grip_threshold={grip_threshold}, pos_scale={pos_scale}, rot_scale={rot_scale}"
        )

    def _spin_ros(self):
        """Background thread to spin ROS2 executor."""
        while self._spinning and rclpy.ok():
            self._executor.spin_once(timeout_sec=0.01)

    def _tf_callback(self, msg: TFMessage):
        """Process TF messages to extract hand poses."""
        now = time.monotonic()
        with self._lock:
            for transform in msg.transforms:
                frame = transform.child_frame_id
                for side, tf_frame in self.TF_FRAMES.items():
                    if frame == tf_frame:
                        t = transform.transform.translation
                        r = transform.transform.rotation
                        pos = np.array([t.x, t.y, t.z], dtype=np.float64)
                        quat = np.array([r.x, r.y, r.z, r.w], dtype=np.float64)
                        self._hand_poses[side] = (pos, quat)
                        self._pose_buf[side].append((now, pos, quat))

    def _interp_pose(self, side: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return pose at (now - LERP_LAG_S) by lerp/slerp through the buffer.

        Smooths bursty Quest arrivals: with lag > max inter-arrival, the
        target time always lies between two known samples, so output rate is
        as steady as the consumer's call rate, not Quest's burst pattern.
        """
        with self._lock:
            buf = list(self._pose_buf[side])
            latest = self._hand_poses.get(side)
        if not buf:
            return latest
        target = time.monotonic() - self.LERP_LAG_S
        # newest <= target → no extrapolation needed only if buffer has older entry
        prev = None
        for entry in buf:
            if entry[0] <= target:
                prev = entry
            else:
                if prev is None:
                    return entry[1], entry[2]   # too new, clamp to oldest
                t0, p0, q0 = prev
                t1, p1, q1 = entry
                u = 0.0 if t1 == t0 else (target - t0) / (t1 - t0)
                u = max(0.0, min(1.0, u))
                pos = p0 + u * (p1 - p0)
                quat = np.asarray(quat_slerp(list(q0), list(q1), u), dtype=np.float64)
                return pos, quat
        # All entries older than target → return latest
        return prev[1], prev[2]

    def _joy_callback(self, msg: Joy):
        """Process joystick messages for axes and buttons."""
        with self._lock:
            self._joy_axes = list(msg.axes) if msg.axes else [0.0] * 8
            self._joy_buttons = list(msg.buttons) if msg.buttons else [0] * 17

    # VR frame is yawed VR_YAW_OFFSET_DEG (CCW around Z) relative to robot.
    # Flip sign if motion goes the wrong direction along the affected axis.
    VR_YAW_OFFSET_DEG: float = -90.0

    @classmethod
    def _yaw_quat(cls) -> np.ndarray:
        """Quaternion that rotates a vector from VR frame to robot frame."""
        a = np.deg2rad(-cls.VR_YAW_OFFSET_DEG) / 2.0
        return np.array([0.0, 0.0, np.sin(a), np.cos(a)], dtype=np.float64)

    def _vr_to_robot_pos_delta(self, delta: np.ndarray) -> np.ndarray:
        """Transform position delta from VR frame to robot frame by undoing
        the yaw offset (rotation around Z)."""
        a = np.deg2rad(-self.VR_YAW_OFFSET_DEG)
        c, s = np.cos(a), np.sin(a)
        x, y, z = delta
        return np.array([c * x - s * y, s * x + c * y, z], dtype=np.float64)

    def _vr_to_robot_quat_delta(self, q: np.ndarray) -> np.ndarray:
        """Conjugate a rotation delta from VR to robot frame: q_r = R q_v R^-1."""
        r = self._yaw_quat()
        r_inv = np.array([-r[0], -r[1], -r[2], r[3]], dtype=np.float64)
        return np.array(quat_multiply(quat_multiply(list(r), list(q)), list(r_inv)),
                        dtype=np.float64)

    def get_action(self) -> Tuple[np.ndarray, List[VRState], bool]:
        """
        Get current action from VR controllers.

        Returns:
            action: 16D array [left_8D, right_8D]
                   Each 8D: [dx, dy, dz, qx, qy, qz, qw, gripper]
                   Position delta in meters, rotation delta as quaternion,
                   gripper state (0=open, 1=closed).
                   Identity rotation delta = [0, 0, 0, 1].
            states: List of VRState for each arm
            any_state_changed: True if any arm's state changed this frame
        """
        with self._lock:
            hand_poses = {k: v for k, v in self._hand_poses.items()}
            joy_axes = self._joy_axes.copy()
            joy_buttons = self._joy_buttons.copy()

        action = np.zeros(16, dtype=np.float64)
        # Initialize rotation deltas to identity quaternion [0,0,0,1]
        action[3:7] = [0.0, 0.0, 0.0, 1.0]   # left arm
        action[11:15] = [0.0, 0.0, 0.0, 1.0]  # right arm
        any_state_changed = False

        for arm_idx, arm_name in enumerate(self.ARM_NAMES):
            # Read grip axis (continuous 0.0 - 1.0)
            grip_idx = self._grip_axes[arm_idx]
            grip_value = joy_axes[grip_idx] if grip_idx < len(joy_axes) else 0.0
            grip_pressed = grip_value > self.grip_threshold

            # Read trigger button (binary)
            trigger_idx = self._trigger_buttons[arm_idx]
            trigger_pressed = bool(joy_buttons[trigger_idx]) if trigger_idx < len(joy_buttons) else False

            # Toggle intervention state on grip press (edge detection)
            grip_just_changed = False
            if grip_pressed and not self._prev_grip[arm_idx]:
                grip_just_changed = True
                if self._arm_states[arm_idx] == VRState.AUTONOMOUS:
                    any_state_changed = True
                    self._arm_states[arm_idx] = VRState.INTERVENING
                    self._engage_time[arm_idx] = time.time()
                    print(f"[VR] >>> {arm_name.upper()} arm: ENGAGED (settling {self._settle_duration}s) <<<")
                    # Reset previous pose to current so first delta is zero
                    self._prev_pose[arm_idx] = self._interp_pose(arm_name)
                else:
                    any_state_changed = True
                    self._arm_states[arm_idx] = VRState.AUTONOMOUS
                    self._engage_time[arm_idx] = None
                    print(f"[VR] >>> {arm_name.upper()} arm: RELEASED (grip) <<<")
                    self._prev_pose[arm_idx] = None
            self._prev_grip[arm_idx] = grip_pressed

            # Trigger button: toggle gripper (edge detection, only while engaged)
            # Skip trigger on the same frame as grip change — squeezing the grip
            # physically curls fingers onto the trigger, causing false toggles.
            if not grip_just_changed and self._arm_states[arm_idx] == VRState.INTERVENING:
                if trigger_pressed and not self._prev_trigger[arm_idx]:
                    self._gripper_states[arm_idx] = 1.0 - self._gripper_states[arm_idx]
                    gripper_status = "CLOSED" if self._gripper_states[arm_idx] > 0.5 else "OPEN"
                    print(f"[VR] {arm_name.upper()} gripper: {gripper_status}")
            self._prev_trigger[arm_idx] = trigger_pressed

            # Compute frame-to-frame delta if engaged
            offset = arm_idx * 8

            if self._arm_states[arm_idx] == VRState.INTERVENING:
                current_pose = self._interp_pose(arm_name)

                # Settle period: output zero deltas while TF stream stabilizes
                settling = (
                    self._engage_time[arm_idx] is not None
                    and time.time() - self._engage_time[arm_idx] < self._settle_duration
                )

                if settling:
                    # Zero deltas, but keep updating _prev_pose so first real
                    # delta after settle is from the stabilized position
                    action[offset+7] = self._gripper_states[arm_idx]
                elif current_pose is not None and self._prev_pose[arm_idx] is not None:
                    curr_pos, curr_quat = current_pose
                    prev_pos, prev_quat = self._prev_pose[arm_idx]

                    # Position delta (frame-to-frame)
                    pos_delta = self._vr_to_robot_pos_delta(curr_pos - prev_pos)

                    # Rotation delta as quaternion: q_delta = q_current * q_prev_inverse
                    q_delta = quat_normalize(quat_multiply(
                        list(curr_quat),
                        quat_inverse(list(prev_quat))
                    ))
                    q_delta = self._vr_to_robot_quat_delta(np.asarray(q_delta))

                    action[offset:offset+3] = pos_delta * self.pos_scale
                    action[offset+3:offset+7] = q_delta
                    action[offset+7] = self._gripper_states[arm_idx]

                # Update previous pose for next frame (always, including during settle)
                if current_pose is not None:
                    self._prev_pose[arm_idx] = current_pose
            else:
                # Not engaged — gripper state still tracked
                action[offset+7] = self._gripper_states[arm_idx]

        return action, self._arm_states.copy(), any_state_changed

    def get_hand_poses(self) -> Dict[str, Optional[Tuple[np.ndarray, np.ndarray]]]:
        """Get current raw hand poses (for debugging)."""
        with self._lock:
            return {k: v for k, v in self._hand_poses.items()}

    def get_joy_data(self) -> Tuple[List[float], List[int]]:
        """Get current raw joystick data (for debugging)."""
        with self._lock:
            return self._joy_axes.copy(), self._joy_buttons.copy()

    def has_data(self) -> bool:
        """Check if we're receiving VR data."""
        with self._lock:
            return any(v is not None for v in self._hand_poses.values())

    def reset(self):
        """Reset all arms to autonomous state."""
        for i in range(self.num_arms):
            self._arm_states[i] = VRState.AUTONOMOUS
            self._gripper_states[i] = 0.0
            self._prev_grip[i] = False
            self._prev_trigger[i] = False
            self._prev_pose[i] = None
            self._engage_time[i] = None
        print("[VR] All arms reset to AUTONOMOUS")

    def close(self):
        """Shutdown ROS2 node and executor."""
        self._spinning = False
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=2.0)
        try:
            self._node.destroy_node()
        except Exception:
            pass
        print("[VR] Node closed")

    def __del__(self):
        self.close()