#!/usr/bin/env python3
"""
Test script for SpaceMouse controlling one or more real robot arms.

Arm count auto-detected from len(--ports). Arm names:
  1 arm  → ["follow"]
  2 arms → ["left", "right"]
  N arms → ["arm0", ..., "arm{N-1}"]

Usage (run from repo root):
    # Dual arm (default ports 50051 50053)
    python Airbot/spacemouse/test_spacemouse_robot.py

    # Single arm
    python Airbot/spacemouse/test_spacemouse_robot.py --ports 50051

    # Dual arm with auto-calibration
    python Airbot/spacemouse/test_spacemouse_robot.py --ports 50051 50053 --auto-calibrate

    # Custom ports
    python Airbot/spacemouse/test_spacemouse_robot.py --ports 50050 50051

    # SpaceMouse only (no robot)
    python Airbot/spacemouse/test_spacemouse_robot.py --no-robot

    # With video recording (USB cameras on device 0 and 2)
    python Airbot/spacemouse/test_spacemouse_robot.py --cameras 0 2

    # With RealSense camera video recording
    python Airbot/spacemouse/test_spacemouse_robot.py --realsense 243222071389

Controls:
    - Left button: Toggle intervention mode for that arm
    - Right button: Toggle gripper (only while intervening)
    - Move SpaceMouse: Control 6DOF pose of corresponding arm
    - ENTER key: Start/stop video recording (only while intervening)
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
    print("[Warning] pyrealsense2 not available. RealSense camera disabled.")
    REALSENSE_AVAILABLE = False

# Add Airbot/ to sys.path so `spacemouse` (this package) and `safe_arm` are findable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spacemouse import (
    DualAirbotSpaceMouse,
    create_dual_spacemouse,
    SpaceMouseState,
)

# Try to import robot arm library
try:
    from airbot_py.arm import AIRBOTArm, RobotMode, SpeedProfile
    from safe_arm import SafeAIRBOTArm, default_z_min
    ROBOT_AVAILABLE = True
except ImportError:
    print("[Warning] airbot_py.arm not available. Robot control disabled.")
    ROBOT_AVAILABLE = False
    SpeedProfile = None


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> List[float]:
    """Convert Euler angles to quaternion [qx, qy, qz, qw]."""
    cr, cp, cy = np.cos(roll/2), np.cos(pitch/2), np.cos(yaw/2)
    sr, sp, sy = np.sin(roll/2), np.sin(pitch/2), np.sin(yaw/2)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return [qx, qy, qz, qw]


def quaternion_to_euler(qx: float, qy: float, qz: float, qw: float) -> Tuple[float, float, float]:
    """Convert quaternion to Euler angles (roll, pitch, yaw)."""
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (qw * qy - qz * qx)
    pitch = np.arcsin(np.clip(sinp, -1, 1))

    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


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


def apply_rotation_delta(current_quat: List[float], droll: float, dpitch: float, dyaw: float) -> List[float]:
    """Apply small euler rotation delta to current quaternion via quaternion multiplication.

    This avoids gimbal lock by never converting the full orientation to euler angles.
    """
    delta_quat = euler_to_quaternion(droll, dpitch, dyaw)
    result = quat_multiply(delta_quat, current_quat)
    # Normalize to prevent drift
    norm = np.sqrt(sum(x*x for x in result))
    return [x / norm for x in result]


class ArmController:
    """Controller for 1 or N robot arms with SpaceMouse input.

    Arm count is inferred from len(ports). Arm names auto-derived:
      1 arm  → ["follow"]
      2 arms → ["left", "right"]
      N arms → ["arm0", "arm1", ..., "arm{N-1}"]
    """

    @staticmethod
    def _default_arm_names(num_arms: int) -> List[str]:
        if num_arms == 1:
            return ["follow"]
        if num_arms == 2:
            return ["left", "right"]
        return [f"arm{i}" for i in range(num_arms)]

    def __init__(
        self,
        ports: List[int] = None,
        robot_url: str = "localhost",
        pos_scale: float = 0.1,
        rot_scale: float = 0.1,
        pos_scale_per_arm: List[float] = None,
        rot_scale_per_arm: List[float] = None,
        auto_calibrate: bool = False,
        use_mock_spacemouse: bool = False,
        no_robot: bool = False,
        camera_devices: List[int] = None,
        realsense_serial: str = None,
        video_output_dir: str = "./videos",
        video_fps: int = 30,
        video_width: int = 640,
        video_height: int = 480,
    ):
        self.ports = list(ports) if ports else [50051, 50053]
        self.num_arms = len(self.ports)
        self.arm_names = self._default_arm_names(self.num_arms)
        # Keep ARM_NAMES as instance attr so existing references keep working.
        self.ARM_NAMES = self.arm_names
        self.robot_url = robot_url
        self.pos_scale = pos_scale
        self.rot_scale = rot_scale
        self.no_robot = no_robot

        # Gripper settings
        self.gripper_open = 0.0
        self.gripper_closed = 0.07

        # Video recording settings
        self.camera_devices = camera_devices or []
        self.realsense_serial = realsense_serial
        self.video_output_dir = Path(video_output_dir)
        self.video_fps = video_fps
        self.video_width = video_width
        self.video_height = video_height

        # Camera and recording state
        self.cameras: List[Tuple[int, "cv2.VideoCapture"]] = []
        self.rs_pipeline = None  # RealSense pipeline
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

        # Initialize SpaceMouse
        print("\n" + "=" * 60)
        print("Initializing Dual SpaceMouse...")
        print("=" * 60)

        self.spacemouse = create_dual_spacemouse(
            num_arms=self.num_arms,
            use_mock=use_mock_spacemouse,
            auto_calibrate=auto_calibrate,
            pos_scale=pos_scale,
            rot_scale=rot_scale,
            pos_scale_per_arm=pos_scale_per_arm,
            rot_scale_per_arm=rot_scale_per_arm,
        )
        print(f"SpaceMouse initialized: {type(self.spacemouse).__name__}")
        if pos_scale_per_arm:
            print(f"  Position scale per arm: {pos_scale_per_arm}")
        if rot_scale_per_arm:
            print(f"  Rotation scale per arm: {rot_scale_per_arm}")

        # Initialize robot arms
        self.arms = {}
        if not no_robot and ROBOT_AVAILABLE:
            print("\n" + "=" * 60)
            print("Connecting to Robot Arms...")
            print("=" * 60)

            for name, port in zip(self.arm_names, self.ports, strict=True):
                print(f"Connecting to {name} arm at {robot_url}:{port}...")
                try:
                    raw_arm = AIRBOTArm(robot_url, port)
                    if raw_arm.connect():
                        arm = SafeAIRBOTArm(raw_arm, z_min=default_z_min(name))
                        self.arms[name] = arm
                        # Set speed profile to FAST for better responsiveness
                        arm.set_speed_profile(SpeedProfile.FAST)
                        info = arm.get_product_info()
                        print(f"  [OK] {name} arm connected: {info.get('product_type', 'unknown')} (FAST mode)")
                    else:
                        print(f"  [FAIL] Could not connect to {name} arm")
                except Exception as e:
                    print(f"  [ERROR] {name} arm: {e}")

        if not self.arms and not no_robot:
            print("\n[WARNING] No robot arms connected. Running in SpaceMouse-only mode.")

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

        if self.cameras:
            print(f"\n  {len(self.cameras)} camera(s) initialized")
        else:
            print("\n  [WARNING] No cameras available for recording")

    def _init_realsense(self):
        """Initialize Intel RealSense camera for video recording."""
        print("\n" + "=" * 60)
        print("Initializing RealSense Camera...")
        print("=" * 60)

        try:
            self.rs_pipeline = rs.pipeline()
            self.rs_config = rs.config()

            # Enable device by serial number
            self.rs_config.enable_device(self.realsense_serial)
            self.rs_config.enable_stream(
                rs.stream.color,
                self.video_width,
                self.video_height,
                rs.format.bgr8,
                self.video_fps
            )

            # Start pipeline to verify connection
            profile = self.rs_pipeline.start(self.rs_config)
            color_stream = profile.get_stream(rs.stream.color)
            color_profile = color_stream.as_video_stream_profile()

            actual_w = color_profile.width()
            actual_h = color_profile.height()
            actual_fps = color_profile.fps()

            print(f"  [OK] RealSense {self.realsense_serial}: {actual_w}x{actual_h} @ {actual_fps:.0f}fps")

            # Update actual values
            self.video_width = actual_w
            self.video_height = actual_h
            self.video_fps = int(actual_fps)

        except Exception as e:
            print(f"  [FAIL] Could not initialize RealSense: {e}")
            self.rs_pipeline = None
            self.rs_config = None

    def _has_camera(self) -> bool:
        """Check if any camera is available."""
        return bool(self.cameras) or (self.rs_pipeline is not None)

    def start_recording(self):
        """Start recording video from all cameras."""
        if not self._has_camera() or self.is_recording:
            return False

        # Create output directory
        self.video_output_dir.mkdir(parents=True, exist_ok=True)

        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create video writers
        self.video_writers = []
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        # RealSense camera
        if self.rs_pipeline is not None:
            output_path = self.video_output_dir / f"realsense_{self.realsense_serial}_{timestamp}.mp4"
            writer = cv2.VideoWriter(str(output_path), fourcc, self.video_fps, (self.video_width, self.video_height))
            if writer.isOpened():
                self.video_writers.append(("realsense", writer, output_path))
                print(f"\n  [REC] RealSense -> {output_path}")
            else:
                print(f"\n  [ERROR] Could not create video writer for RealSense")

        # USB cameras
        for device, cap in self.cameras:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            output_path = self.video_output_dir / f"cam{device}_{timestamp}.mp4"
            writer = cv2.VideoWriter(str(output_path), fourcc, self.video_fps, (w, h))
            if writer.isOpened():
                self.video_writers.append((device, writer, output_path))
                print(f"\n  [REC] Camera {device} -> {output_path}")
            else:
                print(f"\n  [ERROR] Could not create video writer for camera {device}")

        if not self.video_writers:
            return False

        self.is_recording = True
        self.recording_start_time = time.time()
        self._record_stop.clear()

        # Start recording thread
        def _record_loop():
            while not self._record_stop.is_set():
                # RealSense camera
                if self.rs_pipeline is not None:
                    try:
                        frames = self.rs_pipeline.wait_for_frames(timeout_ms=100)
                        color_frame = frames.get_color_frame()
                        if color_frame:
                            frame = np.asanyarray(color_frame.get_data())
                            # Find realsense writer
                            for dev, writer, _ in self.video_writers:
                                if dev == "realsense":
                                    writer.write(frame)
                                    break
                    except Exception:
                        pass

                # USB cameras
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

        # Calculate duration
        duration = time.time() - self.recording_start_time if self.recording_start_time else 0

        # Release video writers
        for device, writer, path in self.video_writers:
            writer.release()
            print(f"\n  [SAVED] Camera {device}: {path} ({duration:.1f}s)")

        self.video_writers = []
        self.is_recording = False
        self.recording_start_time = None

    def reset_to_position(self, reset_actions: List[List[float]]):
        """Reset all arms to specific joint positions.

        Args:
            reset_actions: list of per-arm 7D joint targets, one per arm.
                           Must have length == self.num_arms.
        """
        if not self.arms:
            print("No robot arms to reset")
            return

        if len(reset_actions) != self.num_arms:
            print(f"[ERROR] reset_actions length {len(reset_actions)} != num_arms {self.num_arms}")
            return

        arm_positions = dict(zip(self.arm_names, reset_actions))

        print(f"\nResetting arms to positions:")
        for name, pos in arm_positions.items():
            print(f"  {name.upper():>6}: {[f'{j:.3f}' for j in pos[:6]]}")

        for name, arm in self.arms.items():
            pos = arm_positions.get(name)
            if pos is None:
                continue
            try:
                arm.switch_mode(RobotMode.PLANNING_POS)
                arm.move_to_joint_pos(pos[:6])
                if len(pos) > 6:
                    arm.servo_eef_pos([pos[6]])
                print(f"  [OK] {name} arm reset")
            except Exception as e:
                print(f"  [ERROR] {name} arm reset failed: {e}")

        # Wait for movement to complete
        time.sleep(2)

    def switch_to_servo_mode(self):
        """Switch all arms to servo mode for real-time control."""
        for name, arm in self.arms.items():
            try:
                arm.switch_mode(RobotMode.SERVO_CART_POSE)
                print(f"  [OK] {name} arm in SERVO_CART_POSE mode")
            except Exception as e:
                print(f"  [ERROR] {name} arm mode switch failed: {e}")

    def control_arm(self, name: str, sm_action: np.ndarray):
        """Control a single arm based on SpaceMouse input."""
        if name not in self.arms:
            return

        arm = self.arms[name]

        try:
            # Get current pose
            pose = arm.get_end_pose()
            if pose is None:
                return

            current_pos, current_ori = pose

            # Apply position delta
            new_pos = [
                current_pos[0] + sm_action[0],
                current_pos[1] + sm_action[1],
                current_pos[2] + sm_action[2],
            ]

            # Apply rotation delta directly in quaternion space (avoids gimbal lock)
            new_quat = apply_rotation_delta(
                list(current_ori), sm_action[3], sm_action[4], sm_action[5]
            )

            # Send pose command (convert to native float for gRPC)
            arm.servo_cart_pose(
                [[float(v) for v in new_pos], [float(v) for v in new_quat]]
            )

            # Control gripper
            gripper = self.gripper_closed if sm_action[6] > 0.5 else self.gripper_open
            arm.servo_eef_pos([gripper])

        except Exception as e:
            print(f"\n[ERROR {name}] {e}")

    def get_arm_state(self, name: str) -> Optional[dict]:
        """Get current state of an arm."""
        if name not in self.arms:
            return None

        arm = self.arms[name]
        try:
            pose = arm.get_end_pose()
            joint_pos = arm.get_joint_pos()
            eef_pos = arm.get_eef_pos()
            return {
                "pose": pose,
                "joint_pos": joint_pos,
                "eef_pos": eef_pos,
            }
        except Exception:
            return None

    def run(self, reset_actions: Optional[List[List[float]]] = None):
        """Main control loop.

        Args:
            reset_actions: list of per-arm 7D joint targets, or None to skip reset.
                           Length must equal self.num_arms.
        """
        print("\n" + "=" * 70)
        print(f"SpaceMouse Robot Control ({self.num_arms} arm{'s' if self.num_arms > 1 else ''})")
        print("=" * 70)
        print()
        print("Controls:")
        print("  - Left button:  Toggle intervention mode for that arm")
        print("  - Right button: Toggle gripper (only while intervening)")
        print("  - Move handle:  Control 6DOF pose")
        if self._has_camera():
            print()
            print("Video Recording:")
            print("  - Press ENTER while INTERVENING to start recording")
            print("  - Press ENTER again to stop recording")
        print()
        print("Status display:")
        print("  [INT] = Intervention mode (SpaceMouse controlling)")
        print("  [AUT] = Autonomous mode (SpaceMouse input ignored)")
        if self._has_camera():
            print("  [REC] = Recording in progress")
        print()
        print("Press Ctrl+C to exit")
        print("=" * 70)

        # Reset to initial position if provided
        if reset_actions and self.arms:
            self.reset_to_position(reset_actions)
            self.switch_to_servo_mode()

        # Reset SpaceMouse state
        self.spacemouse.reset()

        # Set up non-blocking keyboard input for Enter key detection
        old_settings = termios.tcgetattr(sys.stdin)

        def check_enter_key():
            """Non-blocking check if Enter was pressed."""
            if select.select([sys.stdin], [], [], 0)[0]:
                char = sys.stdin.read(1)
                return char == '\n' or char == '\r'
            return False

        iteration = 0
        was_intervening = False  # Track previous intervention state
        try:
            tty.setcbreak(sys.stdin.fileno())
            while True:
                # Push current real gripper state every step so driver
                # snaps to it on takeover instead of popping to 0.0 (open).
                for arm_idx, arm in enumerate(self.arms.values()):
                    try:
                        pos = arm.get_eef_pos()
                        if pos:
                            closed = abs(pos[0] - self.gripper_closed) < abs(pos[0] - self.gripper_open)
                            self.spacemouse.update_gripper_state(arm_idx, closed)
                    except Exception:
                        pass

                # Get SpaceMouse action
                action, arm_states, state_changed = self.spacemouse.get_action()

                # Check if any arm is intervening
                any_intervening = any(s == SpaceMouseState.INTERVENING for s in arm_states)

                # Handle Enter key for recording (only when intervening)
                if check_enter_key():
                    if any_intervening and self._has_camera():
                        if self.is_recording:
                            self.stop_recording()
                        else:
                            self.start_recording()

                # Track intervention state change for potential auto-stop
                was_intervening = any_intervening

                # Print state changes
                if state_changed:
                    print(f"\n{'='*50}")
                    for i, name in enumerate(self.ARM_NAMES):
                        state_str = "INTERVENING" if arm_states[i] == SpaceMouseState.INTERVENING else "AUTONOMOUS"
                        print(f"  {name.upper()} arm: {state_str}")
                    print(f"{'='*50}\n")

                # Control arms based on SpaceMouse input
                for i, name in enumerate(self.ARM_NAMES):
                    if arm_states[i] == SpaceMouseState.INTERVENING:
                        offset = i * 7
                        arm_action = action[offset:offset + 7]
                        self.control_arm(name, arm_action)

                # Print status every 10 iterations
                if iteration % 10 == 0:
                    rec_status = ""
                    if self.is_recording:
                        elapsed = time.time() - self.recording_start_time if self.recording_start_time else 0
                        rec_status = f" [REC {elapsed:.1f}s]"

                    parts = []
                    for i, name in enumerate(self.ARM_NAMES):
                        st = "[INT]" if arm_states[i] == SpaceMouseState.INTERVENING else "[AUT]"
                        off = i * 7
                        pos = action[off:off + 3]
                        grip = action[off + 6]
                        parts.append(
                            f"{name.upper()} {st} "
                            f"[{pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f}] "
                            f"G:{grip:.1f}"
                        )
                    print("\r" + " | ".join(parts) + rec_status, end="", flush=True)

                iteration += 1
                time.sleep(0.02)  # 50Hz control loop

        except KeyboardInterrupt:
            print("\n\nStopping...")
        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            self.shutdown()

    def shutdown(self):
        """Clean shutdown."""
        print("\nShutting down...")

        # Stop recording if active
        if self.is_recording:
            self.stop_recording()

        # Close RealSense pipeline
        if self.rs_pipeline is not None:
            try:
                self.rs_pipeline.stop()
                print(f"  RealSense {self.realsense_serial} closed")
            except Exception as e:
                print(f"  RealSense close error: {e}")

        # Close USB cameras
        for device, cap in self.cameras:
            try:
                cap.release()
                print(f"  Camera {device} closed")
            except Exception as e:
                print(f"  Camera {device} close error: {e}")

        # Close SpaceMouse
        try:
            self.spacemouse.close()
            print("  SpaceMouse closed")
        except Exception as e:
            print(f"  SpaceMouse close error: {e}")

        # Disconnect arms
        for name, arm in self.arms.items():
            try:
                arm.disconnect()
                print(f"  {name} arm disconnected")
            except Exception as e:
                print(f"  {name} arm disconnect error: {e}")

        print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Test SpaceMouse controlling 1 or more robot arms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: dual arm with default ports (50051, 50053)
  python Airbot/spacemouse/test_spacemouse_robot.py

  # Single arm
  python Airbot/spacemouse/test_spacemouse_robot.py --ports 50051

  # Dual arm with custom ports and auto-calibration
  python Airbot/spacemouse/test_spacemouse_robot.py --ports 50050 50051 --auto-calibrate

  # Per-arm scales (same length as --ports)
  python Airbot/spacemouse/test_spacemouse_robot.py --ports 50051 50053 --pos-scales 0.5 0.3

  # SpaceMouse only (no robot connection)
  python Airbot/spacemouse/test_spacemouse_robot.py --no-robot
"""
    )

    parser.add_argument(
        "--ports", type=int, nargs="+", default=[50051, 50053],
        help="gRPC ports, one per arm. Length determines num_arms (default: 50051 50053)"
    )
    parser.add_argument(
        "--robot-url", type=str, default="localhost",
        help="Robot URL (default: localhost)"
    )
    parser.add_argument(
        "--pos-scale", type=float, default=0.3,
        help="Shared position scale (default: 0.3; overridden per-arm by --pos-scales)"
    )
    parser.add_argument(
        "--rot-scale", type=float, default=0.3,
        help="Shared rotation scale (default: 0.3; overridden per-arm by --rot-scales)"
    )
    parser.add_argument(
        "--pos-scales", type=float, nargs="+", default=None,
        help="Per-arm position scales (one value per --ports entry). Overrides --pos-scale."
    )
    parser.add_argument(
        "--rot-scales", type=float, nargs="+", default=None,
        help="Per-arm rotation scales (one value per --ports entry). Overrides --rot-scale."
    )
    parser.add_argument(
        "--auto-calibrate", action="store_true",
        help="Run auto-calibration to identify SpaceMouse device assignment"
    )
    parser.add_argument(
        "--mock-spacemouse", action="store_true",
        help="Use mock SpaceMouse (no hardware)"
    )
    parser.add_argument(
        "--no-robot", action="store_true",
        help="Don't connect to robots (SpaceMouse test only)"
    )
    parser.add_argument(
        "--reset-actions", type=float, nargs="+", default=None,
        metavar="VAL",
        help="Flat list of per-arm 7D reset joints: [arm0_j1-j6, arm0_gripper, arm1_...]. "
             "Length must equal num_arms*7. Omit to use built-in default for each arm."
    )
    parser.add_argument(
        "--no-reset", action="store_true",
        help="Skip initial reset to default position"
    )
    parser.add_argument(
        "--cameras", type=int, nargs="+", default=[],
        metavar="DEVICE",
        help="USB camera device numbers for video recording (e.g., --cameras 0 2)"
    )
    parser.add_argument(
        "--realsense", type=str, default=None,
        metavar="SERIAL",
        help="RealSense camera serial number (e.g., --realsense 243222071389)"
    )
    parser.add_argument(
        "--video-dir", type=str, default="./videos",
        help="Output directory for recorded videos (default: ./videos)"
    )
    parser.add_argument(
        "--video-fps", type=int, default=30,
        help="Video recording FPS (default: 30)"
    )
    parser.add_argument(
        "--video-width", type=int, default=640,
        help="Video frame width (default: 640)"
    )
    parser.add_argument(
        "--video-height", type=int, default=480,
        help="Video frame height (default: 480)"
    )

    args = parser.parse_args()
    num_arms = len(args.ports)

    # Validate per-arm scale list lengths
    pos_scale_per_arm = args.pos_scales
    rot_scale_per_arm = args.rot_scales
    if pos_scale_per_arm is not None and len(pos_scale_per_arm) != num_arms:
        parser.error(f"--pos-scales has {len(pos_scale_per_arm)} values, "
                     f"but --ports has {num_arms} arms")
    if rot_scale_per_arm is not None and len(rot_scale_per_arm) != num_arms:
        parser.error(f"--rot-scales has {len(rot_scale_per_arm)} values, "
                     f"but --ports has {num_arms} arms")

    # Build reset_actions: unflatten --reset-actions if given, else use default per arm.
    default_reset = [-0.001618, -1.036, 0.842, -1.615, 0.634, 1.695, 0.0]
    if args.no_reset:
        reset_actions = None
    elif args.reset_actions is None:
        reset_actions = [list(default_reset) for _ in range(num_arms)]
    else:
        if len(args.reset_actions) != num_arms * 7:
            parser.error(f"--reset-actions expects {num_arms * 7} values "
                         f"({num_arms} arms × 7), got {len(args.reset_actions)}")
        reset_actions = [
            list(args.reset_actions[i * 7 : (i + 1) * 7])
            for i in range(num_arms)
        ]

    controller = ArmController(
        ports=args.ports,
        robot_url=args.robot_url,
        pos_scale=args.pos_scale,
        rot_scale=args.rot_scale,
        pos_scale_per_arm=pos_scale_per_arm,
        rot_scale_per_arm=rot_scale_per_arm,
        auto_calibrate=args.auto_calibrate,
        use_mock_spacemouse=args.mock_spacemouse,
        no_robot=args.no_robot,
        camera_devices=args.cameras,
        realsense_serial=args.realsense,
        video_output_dir=args.video_dir,
        video_fps=args.video_fps,
        video_width=args.video_width,
        video_height=args.video_height,
    )

    controller.run(reset_actions=reset_actions)


if __name__ == "__main__":
    main()
