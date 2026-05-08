"""
Example of demonstrating the speed profile of the robot arm.
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


def set_speed_profile():
    """
    Example of demonstrating the speed profile of the robot arm.
    """

    import argparse
    import time

    from airbot_py.arm import AIRBOTPlay, RobotMode, SpeedProfile

    print(
        "\x1b[1;33mThe robot arm is about to be move. Please make sure no obstacles exist in the surrounding environment.\x1b[0m"
    )
    input("\x1b[1;35mPress Enter to continue...\x1b[0m")
    print("\x1b[1;32mPress Ctrl+C to stop...\x1b[0m")

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
        try:
            robot.move_to_joint_pos([0, 0, 0, 0, 0, 0])
            robot.move_to_joint_pos([0, -2, 2, 0, 1, 0])
            robot.move_to_joint_pos([0, 0, 0, 0, 0, 0])
        finally:
            robot.set_speed_profile(SpeedProfile.DEFAULT)


if __name__ == "__main__":
    set_speed_profile()
