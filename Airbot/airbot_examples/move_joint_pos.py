"""
Example of moving the robot to a joint position
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


def move_joint_pos():
    """
    Move the robot to a joint position
    """
    import argparse
    import json

    from airbot_py.arm import AIRBOTPlay, RobotMode, SpeedProfile

    parser = argparse.ArgumentParser(description="Servo example")
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        default="localhost",
        help="Port to connect to the server",
    )
    parser.add_argument(
        "-p", "--port", type=int, default=50051, help="Port to connect to the server"
    )
    parser.add_argument(
        "-j",
        "--joints",
        type=json.loads,
        help="Target joint positions, a 6-float array",
        default=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    parser.add_argument(
        "-e",
        "--end-pos",
        type=float,
        help="Target end effector position, a float",
        default=0.0,
    )
    parser.add_argument(
        "-S",
        "--speed",
        choices=["slow", "fast", "default"],
        default="default",
        help="Speed profile for the robot arm",
    )

    args = parser.parse_args()

    with AIRBOTPlay(url=args.url, port=args.port) as robot:
        if args.speed == "slow":
            robot.set_speed_profile(SpeedProfile.SLOW)
        elif args.speed == "fast":
            robot.set_speed_profile(SpeedProfile.FAST)
        else:
            robot.set_speed_profile(SpeedProfile.DEFAULT)
        robot.switch_mode(RobotMode.PLANNING_POS)
        robot.move_eef_pos([args.end_pos])
        robot.move_to_joint_pos([float(i) for i in args.joints])
        robot.set_speed_profile(SpeedProfile.DEFAULT)


if __name__ == "__main__":
    move_joint_pos()
