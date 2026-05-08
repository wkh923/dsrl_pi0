#!/usr/bin/env python3
"""
Dual-arm robot teleoperation using Quest3 VR controllers.

This script connects Quest3 VR controllers (via ROS2) to two AIRBOT robot arms,
allowing real-time bimanual teleoperation.

Control scheme (hybrid delta mode):
    - Squeeze grip:   Engage control — VR hand movement → robot arm movement
    - Release grip:   Disengage (clutch) — robot holds position, reposition freely
    - Squeeze trigger: Toggle gripper open/close (while engaged)

Usage:
    # Basic usage with default settings
    python test_dual_vr_robot.py

    # Without robot (VR input test only)
    python test_dual_vr_robot.py --no-robot

    # Custom ports and scaling
    python test_dual_vr_robot.py --left-port 50051 --right-port 50053 --pos-scale 1.5

    # Print raw VR data for debugging
    python test_dual_vr_robot.py --no-robot --print-raw

    # With video recording
    python test_dual_vr_robot.py --cameras 0 2
    python test_dual_vr_robot.py --realsense 243222071389
"""

import argparse
import sys
import time
import threading
import select
import termios
import tty
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    print("[Warning] cv2 not available. Video recording disabled.")
    CV2_AVAILABLE = False

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False

from vr_input import DualVRInput, VRState, quat_multiply, quat_normalize, quat_slerp

# Add Airbot/ to sys.path so `safe_arm` resolves to the canonical module.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Try to import robot arm library
try:
    from airbot_py.arm import AIRBOTArm, RobotMode, SpeedProfile
    ROBOT_AVAILABLE = True
except ImportError as e:
    print(f"[Warning] airbot_py.arm not available: {e}. Robot control disabled.")
    ROBOT_AVAILABLE = False
    SpeedProfile = None

# Z-clamp = single source of truth in Airbot/safe_arm.py
from safe_arm import SafeAIRBOTArm, default_z_min


class DualArmVRController:
    """Controller for two robot arms with Quest3 VR controller input."""

    ARM_NAMES = ["left", "right"]

    def __init__(
        self,
        left_port: int = 50051,
        right_port: int = 50053,
        robot_url: str = "localhost",
        pos_scale: float = 5.0,
        rot_scale: float = 5.0,
        grip_threshold: float = 0.5,
        no_robot: bool = False,
        camera_devices: List[int] = None,
        realsense_serial: str = None,
        video_output_dir: str = "./videos",
        video_fps: int = 30,
        video_width: int = 640,
        video_height: int = 480,
        tf_topic: str = "/tf",
        joy_topic: str = "/quest/joystick",
        left_grip_axis: int = 6,
        right_grip_axis: int = 7,
        left_trigger_button: int = 8,
        right_trigger_button: int = 9,
        smooth_alpha: float = 0.4,
        settle_duration: float = 0.5,
        left_z_min: Optional[float] = None,   # None → safe_arm.default_z_min("left")
        right_z_min: Optional[float] = None,  # None → safe_arm.default_z_min("right")
    ):
        self.left_port = left_port
        self.right_port = right_port
        self.robot_url = robot_url
        self.no_robot = no_robot
        self.smooth_alpha = smooth_alpha

        # Per-arm safety z_min — fall back to safe_arm.DEFAULT_Z_MIN_PER_ARM when not overridden
        self._z_min = {
            "left":  left_z_min  if left_z_min  is not None else default_z_min("left"),
            "right": right_z_min if right_z_min is not None else default_z_min("right"),
        }

        # Gripper settings
        self.gripper_open = 0.0
        self.gripper_closed = 0.07

        # Smoothing: per-arm target and commanded pose tracking
        self._target_pos: dict = {}
        self._target_quat: dict = {}
        self._cmd_pos: dict = {}
        self._cmd_quat: dict = {}

        # Video recording settings
        self.camera_devices = camera_devices or []
        self.realsense_serial = realsense_serial
        self.video_output_dir = Path(video_output_dir)
        self.video_fps = video_fps
        self.video_width = video_width
        self.video_height = video_height

        # Camera and recording state
        self.cameras: List[Tuple[int, "cv2.VideoCapture"]] = []
        self.rs_pipeline = None
        self.rs_config = None
        self.video_writers: List["cv2.VideoWriter"] = []
        self.is_recording = False
        self.recording_start_time = None
        self._record_thread: Optional[threading.Thread] = None
        self._record_stop = threading.Event()

        # Initialize RealSense camera (priority over USB cameras)
        if self.realsense_serial and REALSENSE_AVAILABLE and CV2_AVAILABLE:
            self._init_realsense()
        elif self.camera_devices and CV2_AVAILABLE:
            self._init_cameras()

        # Initialize VR input
        print("\n" + "=" * 60)
        print("Initializing Quest3 VR Controllers...")
        print("=" * 60)

        self.vr_input = DualVRInput(
            pos_scale=pos_scale,
            rot_scale=rot_scale,
            grip_threshold=grip_threshold,
            tf_topic=tf_topic,
            joy_topic=joy_topic,
            left_grip_axis=left_grip_axis,
            right_grip_axis=right_grip_axis,
            left_trigger_button=left_trigger_button,
            right_trigger_button=right_trigger_button,
            settle_duration=settle_duration,
        )

        # Wait for VR data
        print("Waiting for Quest3 data...")
        timeout = 15.0
        start = time.time()
        while not self.vr_input.has_data() and time.time() - start < timeout:
            time.sleep(0.1)

        if self.vr_input.has_data():
            print("[OK] Quest3 VR data received")
        else:
            print("[WARNING] No Quest3 data after {:.0f}s. Check ROS2 topics.".format(timeout))

        # Initialize robot arms
        self.arms = {}
        if not no_robot and ROBOT_AVAILABLE:
            print("\n" + "=" * 60)
            print("Connecting to Robot Arms...")
            print("=" * 60)

            for name, port in [("left", left_port), ("right", right_port)]:
                print(f"Connecting to {name} arm at {robot_url}:{port}...")
                try:
                    raw_arm = AIRBOTArm(robot_url, port)
                    if raw_arm.connect():
                        z_min = self._z_min[name]
                        arm = SafeAIRBOTArm(raw_arm, z_min=z_min)
                        self.arms[name] = arm
                        arm.set_speed_profile(SpeedProfile.FAST)
                        info = arm.get_product_info()
                        print(f"  [OK] {name} arm connected: {info.get('product_type', 'unknown')} (FAST mode, z_min={z_min})")
                    else:
                        print(f"  [FAIL] Could not connect to {name} arm")
                except Exception as e:
                    print(f"  [ERROR] {name} arm: {e}")

        if not self.arms and not no_robot:
            print("\n[WARNING] No robot arms connected. Running in VR-only mode.")

    def _init_cameras(self):
        """Initialize USB cameras for video recording."""
        print("\n" + "=" * 60)
        print("Initializing Cameras...")
        print("=" * 60)

        for device in self.camera_devices:
            cap = cv2.VideoCapture(device)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.video_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.video_height)
            cap.set(cv2.CAP_PROP_FPS, self.video_fps)

            if cap.isOpened():
                self.cameras.append((device, cap))
                actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = cap.get(cv2.CAP_PROP_FPS)
                print(f"  [OK] Camera {device}: {actual_w}x{actual_h} @ {actual_fps:.0f}fps")
            else:
                print(f"  [FAIL] Could not open camera {device}")
                cap.release()

    def _init_realsense(self):
        """Initialize Intel RealSense camera for video recording."""
        print("\n" + "=" * 60)
        print("Initializing RealSense Camera...")
        print("=" * 60)

        try:
            self.rs_pipeline = rs.pipeline()
            self.rs_config = rs.config()
            self.rs_config.enable_device(self.realsense_serial)
            self.rs_config.enable_stream(
                rs.stream.color, self.video_width, self.video_height,
                rs.format.bgr8, self.video_fps
            )

            profile = self.rs_pipeline.start(self.rs_config)
            color_stream = profile.get_stream(rs.stream.color)
            color_profile = color_stream.as_video_stream_profile()

            actual_w = color_profile.width()
            actual_h = color_profile.height()
            actual_fps = color_profile.fps()

            print(f"  [OK] RealSense {self.realsense_serial}: {actual_w}x{actual_h} @ {actual_fps:.0f}fps")
            self.video_width = actual_w
            self.video_height = actual_h
            self.video_fps = int(actual_fps)

        except Exception as e:
            print(f"  [FAIL] Could not initialize RealSense: {e}")
            self.rs_pipeline = None
            self.rs_config = None

    def _has_camera(self) -> bool:
        return bool(self.cameras) or (self.rs_pipeline is not None)

    def start_recording(self):
        """Start recording video from all cameras."""
        if not self._has_camera() or self.is_recording:
            return False

        self.video_output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_writers = []
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        if self.rs_pipeline is not None:
            output_path = self.video_output_dir / f"realsense_{self.realsense_serial}_{timestamp}.mp4"
            writer = cv2.VideoWriter(str(output_path), fourcc, self.video_fps, (self.video_width, self.video_height))
            if writer.isOpened():
                self.video_writers.append(("realsense", writer, output_path))
                print(f"\n  [REC] RealSense -> {output_path}")

        for device, cap in self.cameras:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            output_path = self.video_output_dir / f"cam{device}_{timestamp}.mp4"
            writer = cv2.VideoWriter(str(output_path), fourcc, self.video_fps, (w, h))
            if writer.isOpened():
                self.video_writers.append((device, writer, output_path))
                print(f"\n  [REC] Camera {device} -> {output_path}")

        if not self.video_writers:
            return False

        self.is_recording = True
        self.recording_start_time = time.time()
        self._record_stop.clear()

        def _record_loop():
            while not self._record_stop.is_set():
                if self.rs_pipeline is not None:
                    try:
                        frames = self.rs_pipeline.wait_for_frames(timeout_ms=100)
                        color_frame = frames.get_color_frame()
                        if color_frame:
                            frame = np.asanyarray(color_frame.get_data())
                            for dev, writer, _ in self.video_writers:
                                if dev == "realsense":
                                    writer.write(frame)
                                    break
                    except Exception:
                        pass

                for i, (device, cap) in enumerate(self.cameras):
                    ret, frame = cap.read()
                    if ret:
                        for dev, writer, _ in self.video_writers:
                            if dev == device:
                                writer.write(frame)
                                break

                time.sleep(1.0 / self.video_fps)

        self._record_thread = threading.Thread(target=_record_loop, daemon=True)
        self._record_thread.start()
        return True

    def stop_recording(self):
        """Stop recording video."""
        if not self.is_recording:
            return

        self._record_stop.set()
        if self._record_thread is not None:
            self._record_thread.join(timeout=1.0)
        self._record_thread = None

        duration = time.time() - self.recording_start_time if self.recording_start_time else 0
        for device, writer, path in self.video_writers:
            writer.release()
            print(f"\n  [SAVED] Camera {device}: {path} ({duration:.1f}s)")

        self.video_writers = []
        self.is_recording = False
        self.recording_start_time = None

    def reset_to_position(self, joint_pos: List[float], joint_pos_right: List[float] = None):
        """Reset all arms to specific joint positions."""
        if not self.arms:
            print("No robot arms to reset")
            return

        if joint_pos_right is not None:
            left_pos = joint_pos[:7] if len(joint_pos) >= 7 else joint_pos
            right_pos = joint_pos_right[:7] if len(joint_pos_right) >= 7 else joint_pos_right
        elif len(joint_pos) >= 14:
            left_pos = joint_pos[:7]
            right_pos = joint_pos[7:14]
        else:
            left_pos = joint_pos
            right_pos = joint_pos

        arm_positions = {"left": left_pos, "right": right_pos}

        print(f"\nResetting arms to positions:")
        print(f"  LEFT:  {[f'{j:.3f}' for j in left_pos[:6]]}")
        print(f"  RIGHT: {[f'{j:.3f}' for j in right_pos[:6]]}")

        for name, arm in self.arms.items():
            pos = arm_positions.get(name, left_pos)
            try:
                arm.switch_mode(RobotMode.PLANNING_POS)
                arm.move_to_joint_pos(pos[:6])
                if len(pos) > 6:
                    arm.servo_eef_pos([pos[6]])
                print(f"  [OK] {name} arm reset")
            except Exception as e:
                print(f"  [ERROR] {name} arm reset failed: {e}")

        time.sleep(2)

    def switch_to_servo_mode(self):
        """Switch all arms to servo mode for real-time control."""
        for name, arm in self.arms.items():
            try:
                arm.switch_mode(RobotMode.SERVO_CART_POSE)
                print(f"  [OK] {name} arm in SERVO_CART_POSE mode")
            except Exception as e:
                print(f"  [ERROR] {name} arm mode switch failed: {e}")

    def init_arm_tracking(self, name: str):
        """Initialize smoothing targets to current robot pose when engaging."""
        if name not in self.arms:
            return
        pose = self.arms[name].get_end_pose()
        if pose is None:
            return
        self._target_pos[name] = np.array(pose[0], dtype=np.float64)
        self._target_quat[name] = list(pose[1])
        self._cmd_pos[name] = np.array(pose[0], dtype=np.float64)
        self._cmd_quat[name] = list(pose[1])

    def clear_arm_tracking(self, name: str):
        """Clear smoothing state when disengaging."""
        self._target_pos.pop(name, None)
        self._target_quat.pop(name, None)
        self._cmd_pos.pop(name, None)
        self._cmd_quat.pop(name, None)

    def control_arm(self, name: str, vr_action: np.ndarray):
        """Control a single arm based on VR controller input with interpolation.

        Maintains a target pose (accumulated from VR deltas) and smoothly
        interpolates the commanded pose toward it each frame.

        Args:
            name: "left" or "right"
            vr_action: 8D array [dx, dy, dz, qx, qy, qz, qw, gripper]
                       Position delta + quaternion rotation delta + gripper state.
        """
        if name not in self.arms:
            return

        arm = self.arms[name]

        try:
            # Initialize tracking if not yet set (first frame after engage)
            if name not in self._target_pos:
                self.init_arm_tracking(name)
                if name not in self._target_pos:
                    return

            # Accumulate VR delta into target pose
            delta_pos = np.array(vr_action[0:3], dtype=np.float64)
            q_delta = list(vr_action[3:7])

            self._target_pos[name] = self._target_pos[name] + delta_pos
            self._target_quat[name] = quat_normalize(
                quat_multiply(q_delta, self._target_quat[name])
            )

            # Interpolate commanded pose toward target
            alpha = self.smooth_alpha
            self._cmd_pos[name] = self._cmd_pos[name] + alpha * (
                self._target_pos[name] - self._cmd_pos[name]
            )
            self._cmd_quat[name] = quat_slerp(
                self._cmd_quat[name], self._target_quat[name], alpha
            )

            # Send interpolated pose command
            arm.servo_cart_pose(
                [[float(v) for v in self._cmd_pos[name]],
                 [float(v) for v in self._cmd_quat[name]]]
            )

            # Control gripper
            gripper = self.gripper_closed if vr_action[7] > 0.5 else self.gripper_open
            arm.servo_eef_pos([gripper])

        except Exception as e:
            print(f"\n[ERROR {name}] {e}")

    def run(self, reset_action: Optional[List[float]] = None, reset_action_right: Optional[List[float]] = None):
        """Main control loop."""
        # Store reset positions for return-to-home at shutdown
        self._reset_action = reset_action
        self._reset_action_right = reset_action_right
        print("\n" + "=" * 70)
        print("Quest3 VR Dual-Arm Robot Control")
        print("=" * 70)
        print()
        print("Controls:")
        print("  - Squeeze grip:    Engage control for that arm")
        print("  - Release grip:    Disengage (clutch) — reposition freely")
        print("  - Squeeze trigger: Toggle gripper (while engaged)")
        if self._has_camera():
            print()
            print("Video Recording:")
            print("  - Press ENTER while engaged to start recording")
            print("  - Press ENTER again to stop recording")
        print()
        print("Status display:")
        print("  [ENG] = Engaged (VR controlling)")
        print("  [REL] = Released (arm holding position)")
        if self._has_camera():
            print("  [REC] = Recording in progress")
        print()
        print("Press Ctrl+C to exit")
        print("=" * 70)

        # Reset to initial position if provided
        if reset_action and self.arms:
            self.reset_to_position(reset_action, reset_action_right)
            self.switch_to_servo_mode()

        # Reset VR input state
        self.vr_input.reset()

        # Set up non-blocking keyboard input for Enter key detection
        old_settings = termios.tcgetattr(sys.stdin)

        def check_enter_key():
            if select.select([sys.stdin], [], [], 0)[0]:
                char = sys.stdin.read(1)
                return char == '\n' or char == '\r'
            return False

        iteration = 0
        try:
            tty.setcbreak(sys.stdin.fileno())
            while True:
                # Get VR action
                action, arm_states, state_changed = self.vr_input.get_action()

                # Check if any arm is engaged
                any_engaged = any(s == VRState.INTERVENING for s in arm_states)

                # Handle Enter key for recording
                if check_enter_key():
                    if any_engaged and self._has_camera():
                        if self.is_recording:
                            self.stop_recording()
                        else:
                            self.start_recording()

                # Handle state changes — init/clear smoothing targets
                if state_changed:
                    print(f"\n{'='*50}")
                    for i, name in enumerate(self.ARM_NAMES):
                        if arm_states[i] == VRState.INTERVENING:
                            self.init_arm_tracking(name)
                            print(f"  {name.upper()} arm: ENGAGED")
                        else:
                            self.clear_arm_tracking(name)
                            print(f"  {name.upper()} arm: RELEASED")
                    print(f"{'='*50}\n")

                # Control arms based on VR input
                for i, name in enumerate(self.ARM_NAMES):
                    if arm_states[i] == VRState.INTERVENING:
                        offset = i * 8
                        arm_action = action[offset:offset + 8]
                        self.control_arm(name, arm_action)

                # Print status every 10 iterations
                if iteration % 10 == 0:
                    left_state = "[ENG]" if arm_states[0] == VRState.INTERVENING else "[REL]"
                    right_state = "[ENG]" if arm_states[1] == VRState.INTERVENING else "[REL]"

                    left_pos = action[0:3]
                    left_grip = action[7]
                    right_pos = action[8:11]
                    right_grip = action[15]

                    rec_status = ""
                    if self.is_recording:
                        elapsed = time.time() - self.recording_start_time if self.recording_start_time else 0
                        rec_status = f" [REC {elapsed:.1f}s]"

                    status = (
                        f"\rLEFT {left_state} "
                        f"[{left_pos[0]:+.4f},{left_pos[1]:+.4f},{left_pos[2]:+.4f}] "
                        f"G:{left_grip:.1f} | "
                        f"RIGHT {right_state} "
                        f"[{right_pos[0]:+.4f},{right_pos[1]:+.4f},{right_pos[2]:+.4f}] "
                        f"G:{right_grip:.1f}{rec_status}"
                    )
                    print(status, end="", flush=True)

                iteration += 1
                time.sleep(0.02)  # 50Hz control loop

        except KeyboardInterrupt:
            print("\n\nStopping...")
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            self.shutdown()

    def shutdown(self):
        """Clean shutdown."""
        print("\nShutting down...")

        if self.is_recording:
            self.stop_recording()

        if self.rs_pipeline is not None:
            try:
                self.rs_pipeline.stop()
                print(f"  RealSense closed")
            except Exception as e:
                print(f"  RealSense close error: {e}")

        for device, cap in self.cameras:
            try:
                cap.release()
                print(f"  Camera {device} closed")
            except Exception as e:
                print(f"  Camera {device} close error: {e}")

        try:
            self.vr_input.close()
            print("  VR input closed")
        except Exception as e:
            print(f"  VR input close error: {e}")

        # Return arms to home position before disconnecting
        if self.arms and getattr(self, '_reset_action', None) is not None:
            print("  Returning arms to home position...")
            try:
                self.reset_to_position(self._reset_action, self._reset_action_right)
            except Exception as e:
                print(f"  Return to home failed: {e}")

        for name, arm in self.arms.items():
            try:
                arm.disconnect()
                print(f"  {name} arm disconnected")
            except Exception as e:
                print(f"  {name} arm disconnect error: {e}")

        print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Quest3 VR dual-arm robot teleoperation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python test_dual_vr_robot.py

  # VR input only (no robot)
  python test_dual_vr_robot.py --no-robot

  # Custom ports
  python test_dual_vr_robot.py --left-port 50051 --right-port 50053

  # Adjust sensitivity
  python test_dual_vr_robot.py --pos-scale 2.0 --rot-scale 1.5

  # With video recording
  python test_dual_vr_robot.py --cameras 0 2
"""
    )

    # Robot connection
    parser.add_argument("--left-port", type=int, default=50051, help="gRPC port for left arm (default: 50051)")
    parser.add_argument("--right-port", type=int, default=50053, help="gRPC port for right arm (default: 50053)")
    parser.add_argument("--robot-url", type=str, default="localhost", help="Robot URL (default: localhost)")
    parser.add_argument("--no-robot", action="store_true", help="Don't connect to robots (VR test only)")

    # VR control
    parser.add_argument("--pos-scale", type=float, default=5.0, help="Position delta scale (default: 5.0)")
    parser.add_argument("--rot-scale", type=float, default=5.0, help="Rotation delta scale (default: 5.0)")
    parser.add_argument("--grip-threshold", type=float, default=0.5, help="Grip engage threshold 0-1 (default: 0.5)")
    parser.add_argument("--smooth-alpha", type=float, default=0.4, help="Smoothing factor 0-1: lower=smoother, higher=more responsive (default: 0.4)")
    parser.add_argument("--settle-duration", type=float, default=0.5, help="Seconds to hold position after re-engaging grip (default: 0.5)")

    # ROS2 topics
    parser.add_argument("--tf-topic", type=str, default="/tf", help="TF topic (default: /tf)")
    parser.add_argument("--joy-topic", type=str, default="/quest/joystick", help="Joy topic (default: /quest/joystick)")

    # Axis/button mapping (adjust if your Quest3 publisher uses different indices)
    parser.add_argument("--left-grip-axis", type=int, default=6, help="Left grip axis index (default: 6)")
    parser.add_argument("--right-grip-axis", type=int, default=7, help="Right grip axis index (default: 7)")
    parser.add_argument("--left-trigger-button", type=int, default=8, help="Left trigger button index (default: 8)")
    parser.add_argument("--right-trigger-button", type=int, default=9, help="Right trigger button index (default: 9)")

    # Reset position
    parser.add_argument(
        "--reset-action", type=float, nargs=7,
        default=[-0.001618, -1.036, 0.842, -1.615, 0.634, 1.695, 0.0],
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "GRIPPER"),
        help="Initial joint position for LEFT arm"
    )
    parser.add_argument(
        "--reset-action-right", type=float, nargs=7, default=None,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "GRIPPER"),
        help="Initial joint position for RIGHT arm (if different from left)"
    )
    parser.add_argument("--no-reset", action="store_true", help="Skip initial position reset")

    # Per-arm safety z_min — None means fall back to safe_arm.DEFAULT_Z_MIN_PER_ARM
    parser.add_argument("--left-z-min",  type=float, default=None, help="Left arm minimum z height in meters (default: safe_arm.default_z_min('left'))")
    parser.add_argument("--right-z-min", type=float, default=None, help="Right arm minimum z height in meters (default: safe_arm.default_z_min('right'))")

    # Video recording
    parser.add_argument("--cameras", type=int, nargs="+", default=[], metavar="DEVICE",
                        help="USB camera device numbers for video recording")
    parser.add_argument("--realsense", type=str, default=None, metavar="SERIAL",
                        help="RealSense camera serial number")
    parser.add_argument("--video-dir", type=str, default="./videos", help="Video output directory")
    parser.add_argument("--video-fps", type=int, default=30, help="Video FPS (default: 30)")
    parser.add_argument("--video-width", type=int, default=640, help="Video width (default: 640)")
    parser.add_argument("--video-height", type=int, default=480, help="Video height (default: 480)")

    args = parser.parse_args()

    controller = DualArmVRController(
        left_port=args.left_port,
        right_port=args.right_port,
        robot_url=args.robot_url,
        pos_scale=args.pos_scale,
        rot_scale=args.rot_scale,
        grip_threshold=args.grip_threshold,
        smooth_alpha=args.smooth_alpha,
        settle_duration=args.settle_duration,
        no_robot=args.no_robot,
        camera_devices=args.cameras,
        realsense_serial=args.realsense,
        video_output_dir=args.video_dir,
        video_fps=args.video_fps,
        video_width=args.video_width,
        video_height=args.video_height,
        tf_topic=args.tf_topic,
        joy_topic=args.joy_topic,
        left_grip_axis=args.left_grip_axis,
        right_grip_axis=args.right_grip_axis,
        left_trigger_button=args.left_trigger_button,
        right_trigger_button=args.right_trigger_button,
        left_z_min=args.left_z_min,
        right_z_min=args.right_z_min,
    )

    reset_action = None if args.no_reset else args.reset_action
    reset_action_right = None if args.no_reset else args.reset_action_right
    controller.run(reset_action=reset_action, reset_action_right=reset_action_right)


if __name__ == "__main__":
    main()