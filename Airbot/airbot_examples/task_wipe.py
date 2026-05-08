"""
Example application: making the robot arm swing in a sinusoidal motion in cartesian space.
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


def task_wipe():
    """
    Example application: making the robot arm swing in a sinusoidal motion in joint space.
    """

    import argparse
    import json
    import math
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
        "-T",
        "--duration",
        type=float,
        default=2.0,
        help="Duration of the swing in seconds",
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
        robot.move_to_cart_pose([[0.35, 0.0, 0.50], [0, 0, 0, 1]])
        time.sleep(0.5)
        robot.switch_mode(RobotMode.SERVO_JOINT_POS)

        robot.switch_mode(RobotMode.SERVO_CART_POSE)
        start = time.time_ns()
        try:
            while True:
                now = time.time_ns() - start - 0.5 * 1e9
                pose = [
                    [
                        0.35,
                        0.15 * math.sin(now / 1e9 * 2 * math.pi / args.duration),
                        0.50,
                    ],
                    [0, 0, 0, 1],
                ]
                robot.servo_cart_pose(pose)
                robot.servo_eef_pos([0])
                time.sleep(0.01)
        finally:
            robot.set_speed_profile(SpeedProfile.DEFAULT)


if __name__ == "__main__":
    task_wipe()
