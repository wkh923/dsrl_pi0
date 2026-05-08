"""
Make one arm follow the movement of another arm.
"""

import logging

LOG_FORMAT = (
    "[%(asctime)s] [%(levelname)-8s] "
    "[%(name)s.%(funcName)s:%(lineno)d] - %(message)s"
)

# Configure the logging
logging.basicConfig(
    level=logging.DEBUG,  # Set the minimum logging level
    format=LOG_FORMAT,  # Use the custom format
    datefmt="%Y-%m-%d %H:%M:%S",  # Set the date format
    handlers=[logging.StreamHandler()],  # Output logs to stdout (console)
)


def follow():
    """
    Follow the lead robot arm
    """
    import argparse
    import time

    from airbot_py.arm import AIRBOTPlay, RobotMode, SpeedProfile

    parser = argparse.ArgumentParser(description="Switch control mode example")
    parser.add_argument(
        "--lead-url",
        type=str,
        default="localhost",
        help="URL to connect to the lead server",
    )
    parser.add_argument(
        "--follow-url",
        type=str,
        default="localhost",
        help="URL to connect to the follow server",
    )
    parser.add_argument(
        "--lead-port", type=int, default=50051, help="Server port for arm to lead"
    )
    parser.add_argument(
        "--follow-port", type=int, default=50052, help="Server port for arm to follow"
    )

    args = parser.parse_args()
    if args.lead_port == args.follow_port and args.lead_url == args.follow_url:
        raise ValueError(
            "Lead and follow port, lead and follow url cannot be the same at the same time."
        )

    with AIRBOTPlay(
        url=args.lead_url,
        port=args.lead_port,
    ) as robot1, AIRBOTPlay(
        url=args.follow_url,
        port=args.follow_port,
    ) as robot2:
        robot1.switch_mode(RobotMode.PLANNING_POS)
        if (
            sum(
                abs(i - j)
                for i, j in zip(robot1.get_joint_pos(), robot2.get_joint_pos())
            )
            > 0.1
        ):
            robot2.switch_mode(RobotMode.PLANNING_POS)
            robot2.move_to_joint_pos(robot1.get_joint_pos())
            time.sleep(1)
        print("Lead robot arm is ready to follow.")
        robot1.switch_mode(RobotMode.GRAVITY_COMP)
        robot2.switch_mode(RobotMode.SERVO_JOINT_POS)
        robot2.set_speed_profile(SpeedProfile.FAST)
        try:
            while True:
                robot2.servo_joint_pos(robot1.get_joint_pos())
                robot2.servo_eef_pos(robot1.get_eef_pos() or [])
                time.sleep(0.01)
        finally:
            robot1.switch_mode(RobotMode.PLANNING_POS)
            robot2.switch_mode(RobotMode.PLANNING_POS)
            robot2.set_speed_profile(SpeedProfile.DEFAULT)


if __name__ == "__main__":
    follow()
