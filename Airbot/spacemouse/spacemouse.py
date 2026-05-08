"""SpaceMouse input utilities with dual-arm support and auto-calibration."""

from __future__ import annotations

import os
import threading
import time
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import pyspacemouse
    _SPACEMOUSE_AVAILABLE = True
except Exception:
    _SPACEMOUSE_AVAILABLE = False


class SpaceMouseState(Enum):
    AUTONOMOUS = 0
    INTERVENING = 1


class AirbotSpaceMouse:
    def __init__(
        self,
        pos_scale: float = 0.3,
        rot_scale: float = 0.3,
        deadzone: float = 0.001,
    ):
        if not _SPACEMOUSE_AVAILABLE:
            raise RuntimeError("pyspacemouse not installed")
        self.pos_scale = pos_scale
        self.rot_scale = rot_scale
        self.deadzone = deadzone
        self.state = SpaceMouseState.AUTONOMOUS
        self._gripper = 0.0
        # Last known robot gripper state. Consumers should keep pushing this
        # via update_gripper_state() while AUTONOMOUS so on takeover we can
        # snap _gripper to it (avoids the SpaceMouse spuriously commanding
        # the gripper to its stale internal state on the first INTERVENING step).
        self._last_known_closed = False
        self._prev_left = False
        self._prev_right = False
        self._lock = threading.Lock()
        self._latest = {"raw": np.zeros(6, dtype=np.float32), "buttons": [0, 0]}
        self._running = False
        self._device = None
        self._thread = None
        self._connect()

    def _connect(self) -> None:
        self._device = pyspacemouse.open()
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        while self._running:
            try:
                if self._device is None:
                    time.sleep(0.01)
                    continue
                state = self._device.read()
                raw = np.array(
                    [
                        state.y,
                        -state.x,
                        state.z,
                        state.roll,
                        state.pitch,
                        -state.yaw,
                    ],
                    dtype=np.float32,
                )
                with self._lock:
                    self._latest["raw"] = raw
                    self._latest["buttons"] = list(state.buttons)
            except Exception:
                time.sleep(0.05)

    def get_action(self) -> Tuple[np.ndarray, SpaceMouseState, bool]:
        with self._lock:
            raw = self._latest["raw"].copy()
            buttons = list(self._latest["buttons"])

        left = buttons[0] if len(buttons) > 0 else 0
        right = buttons[1] if len(buttons) > 1 else 0
        state_changed = False

        if left and not self._prev_left:
            self.state = (
                SpaceMouseState.INTERVENING
                if self.state == SpaceMouseState.AUTONOMOUS
                else SpaceMouseState.AUTONOMOUS
            )
            state_changed = True
            # Takeover: snap internal gripper to the most recent observed
            # robot state (kept fresh by update_gripper_state from the consumer).
            if self.state == SpaceMouseState.INTERVENING:
                self._gripper = 1.0 if self._last_known_closed else 0.0
        self._prev_left = bool(left)

        if self.state == SpaceMouseState.INTERVENING and right and not self._prev_right:
            self._gripper = 1.0 - self._gripper
        self._prev_right = bool(right)

        raw[np.abs(raw) < self.deadzone] = 0.0
        action = np.zeros(7, dtype=np.float32)
        if self.state == SpaceMouseState.INTERVENING:
            action[:3] = raw[:3] * self.pos_scale
            action[3:6] = raw[3:6] * self.rot_scale
            action[6] = self._gripper
        return action, self.state, state_changed

    @property
    def is_intervening(self) -> bool:
        return self.state == SpaceMouseState.INTERVENING

    def update_gripper_state(self, closed: bool) -> None:
        """Push the robot's current gripper state. Call every step in AUTONOMOUS
        so the driver can snap to it on takeover."""
        self._last_known_closed = bool(closed)

    def reset(self) -> None:
        self.state = SpaceMouseState.AUTONOMOUS
        self._gripper = 0.0
        self._prev_left = False
        self._prev_right = False

    def close(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._device is not None:
            try:
                self._device.close()
            except Exception:
                pass
        self._device = None


class DualAirbotSpaceMouse:
    """Dual SpaceMouse controller with auto-calibration support.

    Supports:
    - Auto-calibration to identify left/right SpaceMouse
    - Per-arm intervention state
    - Device path scanning via /sys/class/hidraw
    - Button edge detection in reader thread (captures presses during inference)
    """
    ARM_NAMES = ["left", "right"]

    def __init__(
        self,
        num_arms: int = 2,
        pos_scale: float = 0.3,
        rot_scale: float = 0.3,
        deadzone: float = 0.001,
        auto_calibrate: bool = False,
        device_paths: Optional[Dict[str, str]] = None,
        pos_scale_per_arm: Optional[List[float]] = None,
        rot_scale_per_arm: Optional[List[float]] = None,
        link_arms: bool = False,  # If True, single button press toggles ALL arms
    ):
        if not _SPACEMOUSE_AVAILABLE:
            raise RuntimeError("pyspacemouse not installed")
        self.num_arms = num_arms
        self.deadzone = deadzone
        self.link_arms = link_arms  # When True, button press on any arm toggles all arms
        # Last known robot gripper state per arm. Consumers should keep
        # pushing this via update_gripper_state() while AUTONOMOUS so the
        # driver can snap _grippers[i] to it on takeover.
        self._last_known_closed = [False for _ in range(num_arms)]

        # Support per-arm scale
        if pos_scale_per_arm is not None and len(pos_scale_per_arm) >= num_arms:
            self.pos_scale_per_arm = pos_scale_per_arm[:num_arms]
        else:
            self.pos_scale_per_arm = [pos_scale] * num_arms
        if rot_scale_per_arm is not None and len(rot_scale_per_arm) >= num_arms:
            self.rot_scale_per_arm = rot_scale_per_arm[:num_arms]
        else:
            self.rot_scale_per_arm = [rot_scale] * num_arms

        # Keep global scale for backward compatibility
        self.pos_scale = pos_scale
        self.rot_scale = rot_scale

        self._states = [SpaceMouseState.AUTONOMOUS for _ in range(num_arms)]
        self._grippers = [0.0 for _ in range(num_arms)]
        self._prev_left = [False for _ in range(num_arms)]
        self._prev_right = [False for _ in range(num_arms)]
        self._lock = threading.Lock()
        self._latest = [
            {"raw": np.zeros(6, dtype=np.float32), "buttons": [0, 0]}
            for _ in range(num_arms)
        ]
        # Pending toggle flags - set by reader thread, consumed by get_action()
        # This captures button presses even during long inference gaps
        self._pending_left_toggle = [False for _ in range(num_arms)]
        self._pending_right_toggle = [False for _ in range(num_arms)]
        self._reader_prev_left = [False for _ in range(num_arms)]
        self._reader_prev_right = [False for _ in range(num_arms)]

        self._running = False
        self._devices: List[Optional[object]] = [None for _ in range(num_arms)]
        self._threads: List[Optional[threading.Thread]] = [None for _ in range(num_arms)]
        self._arm_device_map: Dict[int, int] = {}

        self._connect(device_paths, auto_calibrate)

    def _find_spacemouse_paths(self) -> List[str]:
        """Find all SpaceMouse device paths by scanning /sys/class/hidraw."""
        spacemouse_paths = []
        hidraw_dir = "/sys/class/hidraw"
        VENDOR_ID = "256F"  # 3Dconnexion vendor ID

        if not os.path.exists(hidraw_dir):
            return spacemouse_paths

        for hidraw in sorted(os.listdir(hidraw_dir)):
            uevent_path = os.path.join(hidraw_dir, hidraw, "device/uevent")
            if os.path.exists(uevent_path):
                try:
                    with open(uevent_path, 'r') as f:
                        content = f.read()
                        if VENDOR_ID.lower() in content.lower():
                            device_path = f"/dev/{hidraw}"
                            spacemouse_paths.append(device_path)
                            print(f"[DualSpaceMouse] Found SpaceMouse at {device_path}")
                except Exception:
                    pass

        return spacemouse_paths

    def _connect(self, device_paths: Optional[Dict[str, str]], auto_calibrate: bool) -> None:
        self._running = True

        if self.num_arms == 1:
            self._connect_single_device()
        elif device_paths is not None:
            self._connect_with_paths(device_paths)
        elif auto_calibrate:
            self._run_auto_calibrate()
        else:
            self._connect_multiple_devices()

    def _connect_single_device(self) -> None:
        try:
            device = pyspacemouse.open()
            if device is not None:
                self._devices[0] = device
                self._arm_device_map[0] = 0
                print("[DualSpaceMouse] Single device connected for arm 0 (left)")
                self._start_reader_thread(0)
        except Exception as e:
            print(f"[DualSpaceMouse] Connection error: {e}")

    def _connect_with_paths(self, device_paths: Dict[str, str]) -> None:
        for arm_idx, arm_name in enumerate(self.ARM_NAMES[:self.num_arms]):
            path = device_paths.get(arm_name)
            if path is None:
                continue
            try:
                device = pyspacemouse.open_by_path(path)
                if device is not None:
                    self._devices[arm_idx] = device
                    self._arm_device_map[arm_idx] = arm_idx
                    print(f"[DualSpaceMouse] Device connected for {arm_name} arm: {path}")
                    self._start_reader_thread(arm_idx)
            except (TypeError, AttributeError):
                print("[DualSpaceMouse] Path-based opening not supported, using index-based")
                self._connect_multiple_devices()
                return
            except Exception as e:
                print(f"[DualSpaceMouse] Error opening {path}: {e}")

    def _connect_multiple_devices(self) -> None:
        devices_found = []
        spacemouse_paths = self._find_spacemouse_paths()

        if len(spacemouse_paths) >= self.num_arms:
            for idx, path in enumerate(spacemouse_paths[:self.num_arms]):
                try:
                    device = pyspacemouse.open_by_path(path)
                    if device is not None:
                        devices_found.append((idx, device))
                        print(f"[DualSpaceMouse] Opened device {idx} at {path}")
                except Exception:
                    try:
                        device = pyspacemouse.open(device_index=idx)
                        if device is not None:
                            devices_found.append((idx, device))
                    except Exception:
                        pass
        else:
            for idx in range(self.num_arms):
                try:
                    device = pyspacemouse.open(device_index=idx)
                    if device is not None:
                        devices_found.append((idx, device))
                        print(f"[DualSpaceMouse] Opened device {idx}")
                except Exception as e:
                    print(f"[DualSpaceMouse] Failed to open device {idx}: {e}")

        if len(devices_found) < self.num_arms:
            print(f"[DualSpaceMouse] Warning: Only {len(devices_found)}/{self.num_arms} devices found")

        for arm_idx, (dev_idx, device) in enumerate(devices_found):
            self._devices[arm_idx] = device
            self._arm_device_map[dev_idx] = arm_idx
            self._start_reader_thread(arm_idx)

    def _run_auto_calibrate(self) -> None:
        print("\n" + "="*60)
        print("[DualSpaceMouse] AUTO-CALIBRATION MODE")
        print("="*60)

        self._connect_multiple_devices()

        if sum(1 for d in self._devices if d is not None) < 2:
            print("[DualSpaceMouse] Not enough devices for calibration, using default mapping")
            return

        time.sleep(0.5)

        for arm_idx, arm_name in enumerate(self.ARM_NAMES[:self.num_arms]):
            print(f"\n>>> Move the {arm_name.upper()} arm SpaceMouse now! <<<")
            detected_device = self._detect_movement(timeout=10.0)
            if detected_device is not None:
                self._arm_device_map[detected_device] = arm_idx
                print(f"    [OK] Device {detected_device} assigned to {arm_name} arm")
            else:
                print(f"    [WARN] No movement detected for {arm_name} arm")

        print("\n" + "="*60)
        print(f"[DualSpaceMouse] Calibration complete! Mapping: {self._arm_device_map}")
        print("="*60 + "\n")

    def _detect_movement(self, timeout: float = 10.0) -> Optional[int]:
        start_time = time.time()
        threshold = 0.3

        while time.time() - start_time < timeout:
            with self._lock:
                for dev_idx in range(len(self._devices)):
                    if self._devices[dev_idx] is None:
                        continue
                    if dev_idx in self._arm_device_map:
                        continue
                    data = self._latest[dev_idx]
                    movement = np.linalg.norm(data["raw"])
                    if movement > threshold:
                        return dev_idx
            time.sleep(0.05)
        return None

    def _start_reader_thread(self, arm_idx: int) -> None:
        if self._threads[arm_idx] is not None:
            return
        thread = threading.Thread(target=self._reader, args=(arm_idx,), daemon=True)
        thread.start()
        self._threads[arm_idx] = thread

    def _reader(self, idx: int) -> None:
        """Reader thread that captures SpaceMouse input continuously.

        Button edge detection happens here so presses during inference aren't missed.
        """
        device = self._devices[idx]
        while self._running and device is not None:
            try:
                state = device.read()
                raw = np.array(
                    [
                        state.y,
                        -state.x,
                        state.z,
                        state.roll,
                        state.pitch,
                        -state.yaw,
                    ],
                    dtype=np.float32,
                )

                # Get button states
                buttons = list(state.buttons)
                left = buttons[0] if len(buttons) > 0 else 0
                right = buttons[1] if len(buttons) > 1 else 0

                with self._lock:
                    self._latest[idx]["raw"] = raw
                    self._latest[idx]["buttons"] = buttons

                    # Detect left button rising edge (press) in reader thread
                    # This ensures button presses are captured even during long inference
                    if left and not self._reader_prev_left[idx]:
                        self._pending_left_toggle[idx] = True
                        arm_name = self.ARM_NAMES[idx] if idx < len(self.ARM_NAMES) else f"arm{idx}"
                        print(f"[DualSpaceMouse] Button press detected for {arm_name} (will toggle on next get_action)")
                    self._reader_prev_left[idx] = bool(left)

                    # Detect right button rising edge
                    if right and not self._reader_prev_right[idx]:
                        self._pending_right_toggle[idx] = True
                    self._reader_prev_right[idx] = bool(right)

            except Exception:
                time.sleep(0.05)

    def get_action(self) -> Tuple[np.ndarray, List[SpaceMouseState], bool]:
        """Get current SpaceMouse action and states.

        Button toggles are captured by the reader thread and consumed here.
        This ensures button presses during long inference gaps aren't missed.

        If link_arms=True, a button press on ANY arm toggles ALL arms together.
        """
        action = np.zeros(7 * self.num_arms, dtype=np.float32)
        any_changed = False

        with self._lock:
            latest = [dict(raw=v["raw"].copy(), buttons=list(v["buttons"])) for v in self._latest]
            # Consume pending toggles (set by reader thread)
            pending_left = list(self._pending_left_toggle)
            pending_right = list(self._pending_right_toggle)
            # Clear pending flags
            for i in range(self.num_arms):
                self._pending_left_toggle[i] = False
                self._pending_right_toggle[i] = False

        # If link_arms is enabled, link LEFT button (intervention mode) across all arms
        # But keep RIGHT button (gripper) independent per arm for separate control
        if self.link_arms:
            any_left_toggle = any(pending_left)
            if any_left_toggle:
                pending_left = [True] * self.num_arms
            # Note: pending_right is NOT linked - each arm's gripper is controlled independently

        for arm_idx in range(self.num_arms):
            raw = latest[arm_idx]["raw"]

            # Process left button toggle (intervention mode toggle)
            if pending_left[arm_idx]:
                arm_name = self.ARM_NAMES[arm_idx] if arm_idx < len(self.ARM_NAMES) else f"arm{arm_idx}"
                old_state = self._states[arm_idx]
                self._states[arm_idx] = (
                    SpaceMouseState.INTERVENING
                    if self._states[arm_idx] == SpaceMouseState.AUTONOMOUS
                    else SpaceMouseState.AUTONOMOUS
                )
                state_str = "INTERVENING" if self._states[arm_idx] == SpaceMouseState.INTERVENING else "AUTONOMOUS"
                print(f"[DualSpaceMouse] >>> {arm_name.upper()} arm: {old_state.name} -> {state_str} <<<")
                any_changed = True
                # Takeover: snap to the most recent observed gripper state
                # (pushed by the consumer via update_gripper_state()).
                if self._states[arm_idx] == SpaceMouseState.INTERVENING:
                    self._grippers[arm_idx] = 1.0 if self._last_known_closed[arm_idx] else 0.0

            # Process right button toggle (gripper toggle, only when intervening)
            if self._states[arm_idx] == SpaceMouseState.INTERVENING and pending_right[arm_idx]:
                self._grippers[arm_idx] = 1.0 - self._grippers[arm_idx]
                arm_name = self.ARM_NAMES[arm_idx] if arm_idx < len(self.ARM_NAMES) else f"arm{arm_idx}"
                gripper_status = "CLOSED" if self._grippers[arm_idx] > 0.5 else "OPEN"
                print(f"[DualSpaceMouse] {arm_name.upper()} gripper: {gripper_status}")

            # Apply deadzone and compute action
            raw = np.where(np.abs(raw) < self.deadzone, 0.0, raw)
            arm_action = np.zeros(7, dtype=np.float32)
            if self._states[arm_idx] == SpaceMouseState.INTERVENING:
                pos_scale = self.pos_scale_per_arm[arm_idx]
                rot_scale = self.rot_scale_per_arm[arm_idx]
                arm_action[:3] = raw[:3] * pos_scale
                arm_action[3:6] = raw[3:6] * rot_scale
                arm_action[6] = self._grippers[arm_idx]
            start = arm_idx * 7
            action[start : start + 7] = arm_action

        return action, list(self._states), any_changed

    def get_arm_state(self, arm_idx: int) -> SpaceMouseState:
        if 0 <= arm_idx < self.num_arms:
            return self._states[arm_idx]
        return SpaceMouseState.AUTONOMOUS

    def is_arm_intervening(self, arm_idx: int) -> bool:
        return self.get_arm_state(arm_idx) == SpaceMouseState.INTERVENING

    def update_gripper_state(self, arm_idx: int, closed: bool) -> None:
        """Push the robot's current gripper state for one arm. Call every step
        in AUTONOMOUS so the driver can snap to it on takeover."""
        if 0 <= arm_idx < self.num_arms:
            self._last_known_closed[arm_idx] = bool(closed)

    @property
    def is_any_intervening(self) -> bool:
        return any(s == SpaceMouseState.INTERVENING for s in self._states)

    @property
    def is_all_intervening(self) -> bool:
        return all(s == SpaceMouseState.INTERVENING for s in self._states)

    def reset(self) -> None:
        with self._lock:
            for i in range(self.num_arms):
                self._states[i] = SpaceMouseState.AUTONOMOUS
                self._grippers[i] = 0.0
                self._prev_left[i] = False
                self._prev_right[i] = False
                self._pending_left_toggle[i] = False
                self._pending_right_toggle[i] = False
                self._reader_prev_left[i] = False
                self._reader_prev_right[i] = False
        print("[DualSpaceMouse] All arms reset to AUTONOMOUS")

    def close(self) -> None:
        self._running = False
        for t in self._threads:
            if t is not None:
                t.join(timeout=1.0)
        for dev in self._devices:
            if dev is not None:
                try:
                    dev.close()
                except Exception:
                    pass
        self._devices = [None] * self.num_arms
        self._threads = [None] * self.num_arms

    @property
    def is_intervening(self) -> bool:
        return self.is_any_intervening


class MockSpaceMouse:
    def __init__(self, **kwargs):
        self.state = SpaceMouseState.AUTONOMOUS
        self._action = np.zeros(7, dtype=np.float32)
        self._gripper = 0.0

    def get_action(self) -> Tuple[np.ndarray, SpaceMouseState, bool]:
        return self._action.copy(), self.state, False

    def reset(self) -> None:
        self.state = SpaceMouseState.AUTONOMOUS
        self._gripper = 0.0
        self._action = np.zeros(7, dtype=np.float32)

    def close(self) -> None:
        return None


class MockDualSpaceMouse:
    ARM_NAMES = ["left", "right"]

    def __init__(self, num_arms: int = 2, **kwargs):
        self.num_arms = num_arms
        self._states = [SpaceMouseState.AUTONOMOUS for _ in range(num_arms)]
        self._actions = [np.zeros(7, dtype=np.float32) for _ in range(num_arms)]

    def get_action(self) -> Tuple[np.ndarray, List[SpaceMouseState], bool]:
        return np.concatenate(self._actions[: self.num_arms]), list(self._states), False

    def reset(self) -> None:
        for i in range(self.num_arms):
            self._states[i] = SpaceMouseState.AUTONOMOUS
            self._actions[i] = np.zeros(7, dtype=np.float32)

    def close(self) -> None:
        return None

    @property
    def is_intervening(self) -> bool:
        return any(state == SpaceMouseState.INTERVENING for state in self._states)


def create_spacemouse(use_mock: bool = False, **kwargs):
    if use_mock:
        return MockSpaceMouse(**kwargs)
    return AirbotSpaceMouse(**kwargs)


def create_dual_spacemouse(
    num_arms: int = 2,
    use_mock: bool = False,
    auto_calibrate: bool = False,
    device_paths: Optional[Dict[str, str]] = None,
    link_arms: bool = False,
    **kwargs,
):
    """Create dual SpaceMouse instance.

    Args:
        num_arms: Number of arms (1 or 2)
        use_mock: If True, create MockDualSpaceMouse for testing
        auto_calibrate: If True, run auto-calibration to identify left/right
        device_paths: Dict mapping arm names to device paths
        link_arms: If True, button press on any SpaceMouse toggles ALL arms
        **kwargs: Additional arguments (pos_scale, rot_scale, deadzone, etc.)

    Returns:
        DualAirbotSpaceMouse or MockDualSpaceMouse instance
    """
    if use_mock:
        return MockDualSpaceMouse(num_arms=num_arms, **kwargs)
    return DualAirbotSpaceMouse(
        num_arms=num_arms,
        auto_calibrate=auto_calibrate,
        device_paths=device_paths,
        link_arms=link_arms,
        **kwargs,
    )
