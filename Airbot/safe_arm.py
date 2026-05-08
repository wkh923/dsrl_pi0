"""
Safety wrapper / helpers for AIRBOTArm SDK — single source of truth.

Two clamp paths, both pinning the end-effector z to ``z >= z_min`` (base frame):

* **Cartesian clamp** (default, always on)
    Catches ``servo_cart_pose([[x,y,z], [qx,qy,qz,qw]])``. Used by SpaceMouse /
    VR teleop and SERL training, where the policy/operator sends Cartesian
    poses directly. Cheap, deterministic, no kinematics required.

* **Joint-space clamp** (opt-in, ``clamp_joint_space=True``)
    Catches ``servo_joint_pos`` and ``move_to_joint_pos``. FK predicts the
    end-effector z; if below ``z_min``, IK rebounds T[2,3]=z_min while
    preserving x/y/orientation, returning the corrected joint targets. Used
    by VLA / π0 inference where the policy outputs joint angles. Requires an
    ``arm_kdl`` instance (``airbot_kdl.arm_kdl.ArmKdl``).

Per-arm limits: instantiate one ``SafeAIRBOTArm`` per physical arm with that
arm's ``z_min``. Default ``z_min`` is 2 mm; callers should override.

Two usage patterns:

1. **Wrapper** (preferred when you control AIRBOTArm construction)::

       from safe_arm import SafeAIRBOTArm
       arm = SafeAIRBOTArm(AIRBOTArm(url, port), z_min=0.05)
       arm.servo_cart_pose([...])    # auto-clamped
       arm.get_joint_pos()           # transparently forwarded

2. **Standalone helpers** (when the SDK call site is wrapped by another
   framework that caches bound methods, e.g. ``data_collection.AIRBOTPlay``)::

       from safe_arm import clamp_cart_pose, clamp_joint_pos, SafetyCounter
       counter = SafetyCounter("SafeArm[left]")
       joints = clamp_joint_pos(action[:7], z_min, arm_kdl, counter)
       operator.send_action(joints)
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import numpy as np

_logger = logging.getLogger(__name__)


# Default minimum z height (meters, base frame). Used as a fallback when an
# arm name is not in ``DEFAULT_Z_MIN_PER_ARM``.
DEFAULT_Z_MIN: float = 0.002

# Physical defaults for this machine. Edit these once after calibrating your
# table / mount; all consumers (play_operator, test_inference_with_spacemouse,
# airbot_inference_sync, …) read from here so there is one source of truth.
# Keys must match the arm names used by callers (RobotConfig.robot_groups
# is typically ["left", "right"]; single-arm setups use ["follow"]).
DEFAULT_Z_MIN_PER_ARM: dict[str, float] = {
    "follow": 0.002,
    "left":   0.002,
    "right":  0.002,
}


def default_z_min(arm_name: str) -> float:
    """Look up the per-arm safety floor; fall back to ``DEFAULT_Z_MIN``."""
    return DEFAULT_Z_MIN_PER_ARM.get(arm_name, DEFAULT_Z_MIN)


# ---------------------------------------------------------------------------
# Rate-limited logging helper
# ---------------------------------------------------------------------------


class SafetyCounter:
    """Counts clamp events; ``should_log()`` rate-limits warning spam.

    First 10 events log every time, then every 100th event. Each instance is
    independent — give one to each call site (e.g. one per arm).
    """

    def __init__(self, label: str = "SafeArm") -> None:
        self.label = label
        self.count = 0

    def should_log(self) -> bool:
        self.count += 1
        return self.count <= 10 or self.count % 100 == 0


# ---------------------------------------------------------------------------
# Standalone clamp helpers (also used by the wrapper class below)
# ---------------------------------------------------------------------------


def clamp_cart_pose(
    pose: Sequence,
    z_min: float,
    counter: Optional[SafetyCounter] = None,
):
    """Clamp the z component of ``[[x,y,z], [qx,qy,qz,qw]]`` to ``>= z_min``.

    Returns a new pose list (does not mutate input).
    """
    pos = list(pose[0])
    if pos[2] < z_min:
        label = counter.label if counter is not None else "SafeArm cart"
        count_str = f" (count: {counter.count + 1})" if counter is not None else ""
        if counter is None or counter.should_log():
            _logger.warning(
                f"[{label}] z={pos[2]:.4f} < z_min={z_min:.4f}, clamping{count_str}"
            )
        pos[2] = z_min
        return [pos, list(pose[1])]
    return pose


def clamp_joint_pos(
    joint_pos: Sequence[float],
    z_min: float,
    arm_kdl,
    counter: Optional[SafetyCounter] = None,
) -> List[float]:
    """FK joint_pos[:6] → end-effector z; if ``z < z_min`` IK rebound.

    Args:
        joint_pos: ``[j1..j6]`` (6-element) or ``[j1..j6, gripper, ...]`` (7+).
            Trailing items past index 6 (e.g. gripper) are preserved as-is.
        z_min: minimum allowed end-effector z (meters, base frame).
        arm_kdl: an ``airbot_kdl.arm_kdl.ArmKdl`` instance for FK/IK.
        counter: optional rate-limited logger.

    Returns:
        A new list of the same length as ``joint_pos``. If the original z is
        already safe or IK fails, the input is returned unchanged.
    """
    q = np.asarray(joint_pos[:6], dtype=np.float64)
    T = arm_kdl.forward_kinematics(q)
    z = float(T[2, 3])
    if z >= z_min:
        return list(joint_pos)

    label = counter.label if counter is not None else "SafeArm joint"
    count_str = f" (count: {counter.count + 1})" if counter is not None else ""
    log = counter is None or counter.should_log()
    if log:
        _logger.warning(
            f"[{label}] fk_z={z:.4f} < z_min={z_min:.4f}, IK correcting{count_str}"
        )

    T_corrected = T.copy()
    T_corrected[2, 3] = z_min
    solutions = arm_kdl.inverse_kinematics(
        T_corrected, ref_pos=q, force_calculate=True, use_clip=True
    )
    if not solutions:
        if log:
            _logger.warning(f"[{label}] IK failed, returning original joints")
        return list(joint_pos)

    corrected = list(solutions[0])
    if log:
        T_v = arm_kdl.forward_kinematics(np.asarray(corrected))
        _logger.warning(
            f"[{label}] corrected z={float(T_v[2, 3]):.4f} (target: {z_min:.4f})"
        )

    # Preserve trailing items past index 6 (e.g. gripper).
    if len(joint_pos) > 6:
        corrected.extend(list(joint_pos[6:]))
    return corrected


# ---------------------------------------------------------------------------
# Wrapper class
# ---------------------------------------------------------------------------


class SafeAIRBOTArm:
    """Safety wrapper around an ``AIRBOTArm`` instance.

    See module docstring for the two usage patterns. ``__getattr__`` forwards
    every method other than the explicitly intercepted ones.

    Args:
        arm: the underlying ``AIRBOTArm`` instance (or any object with the
            same servo/move API).
        z_min: minimum allowed end-effector z (meters, base frame). Defaults
            to ``DEFAULT_Z_MIN`` (2 mm); callers should override per arm.
        clamp_joint_space: if True, also intercept ``servo_joint_pos`` and
            ``move_to_joint_pos`` and rebound via FK+IK.
        arm_kdl: required when ``clamp_joint_space=True``.
    """

    def __init__(
        self,
        arm,
        z_min: float = DEFAULT_Z_MIN,
        clamp_joint_space: bool = False,
        arm_kdl=None,
    ) -> None:
        if clamp_joint_space and arm_kdl is None:
            raise ValueError(
                "arm_kdl is required when clamp_joint_space=True. "
                "Pass an airbot_kdl.arm_kdl.ArmKdl instance."
            )
        self._arm = arm
        self._z_min = z_min
        self._clamp_joint_space = clamp_joint_space
        self._arm_kdl = arm_kdl
        self._cart_counter = SafetyCounter(f"SafeArm cart")
        self._joint_counter = SafetyCounter(f"SafeArm joint")

    # --- Cartesian intercept (always on) ---

    def servo_cart_pose(self, pose):
        """Send cartesian pose with z clamped to ``>= z_min``."""
        clamped = clamp_cart_pose(pose, self._z_min, self._cart_counter)
        return self._arm.servo_cart_pose(clamped)

    # --- Joint-space intercepts (opt-in) ---

    def servo_joint_pos(self, joint_pos):
        """Send joint positions, optionally rebounded via FK+IK."""
        if self._clamp_joint_space:
            joint_pos = clamp_joint_pos(
                joint_pos, self._z_min, self._arm_kdl, self._joint_counter
            )
        return self._arm.servo_joint_pos(joint_pos)

    def move_to_joint_pos(self, joint_pos, *args, **kwargs):
        """Plan to joint positions, optionally rebounded via FK+IK."""
        if self._clamp_joint_space:
            joint_pos = clamp_joint_pos(
                joint_pos, self._z_min, self._arm_kdl, self._joint_counter
            )
        return self._arm.move_to_joint_pos(joint_pos, *args, **kwargs)

    # --- Transparent forwarding for everything else ---

    def __getattr__(self, name):
        return getattr(self._arm, name)
