"""
Example of sending continuous cartesian pose commands to the robot arm.
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


def servo_cart_pose():
    """
    Example of sending continuous cartesian pose commands to the robot arm.
    """

    import argparse
    import json
    import time

    print(
        "\x1b[1;33mThe robot arm is about to be move. Please make sure no obstacles exist in the surrounding environment.\x1b[0m"
    )
    input("\x1b[1;35mPress Enter to continue...\x1b[0m")
    print("\x1b[1;32mPress Ctrl+C to stop...\x1b[0m")

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
        "-t",
        "--translation",
        type=json.loads,
        help="Target translation in the cartesian space, a 3-float array",
        default=[0.45, 0.20, 0.45],
    )
    parser.add_argument(
        "-r",
        "--orientation",
        type=json.loads,
        help="Target orientation (in quarternion) in the cartesian space, a 4-float array",
        default=[0.0, 0.0, 0.0, 1.0],
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
        robot.switch_mode(RobotMode.SERVO_CART_POSE)
        try:
            while True:
                robot.servo_cart_pose([args.translation, args.orientation])
                robot.servo_eef_pos([args.end_pos])
                time.sleep(0.01)
        finally:
            robot.set_speed_profile(SpeedProfile.DEFAULT)


if __name__ == "__main__":
    servo_cart_pose()
