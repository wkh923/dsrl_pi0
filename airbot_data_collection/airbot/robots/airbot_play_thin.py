"""Minimal stubs for airbot_play.py fallback when airbot_py (gRPC) is not installed.

Only provides RobotMode, SpeedProfile, and AIRBOTArm type stubs so the module
can be imported without the full airbot_hardware_py CAN-bus backend.
For DSRL, the gRPC backend (airbot_py) should always be used.
"""


class SpeedProfile:
    FAST = "fast"
    SLOW = "slow"


class RobotMode:
    GRAVITY_COMP = "gravity_comp"
    PLANNING_POS = "planning_pos"
    SERVO_JOINT_POS = "servo_joint_pos"
    SERVO_CART_POSE = "servo_cart_pose"


class AIRBOTArm:
    def __init__(self, **kwargs):
        raise NotImplementedError(
            "The thin (CAN-bus) backend is not supported in dsrl_pi0. "
            "Install airbot_py for gRPC support: pip install airbot_py"
        )
