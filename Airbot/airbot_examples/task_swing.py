"""
Example application: making the robot arm swing in a sinusoidal motion in joint space.
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


def task_swing():
    """
    Make the robot arm swing in a sinusoidal motion in joint space.
    """

    import argparse
    import math
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
        "-T",
        "--duration",
        type=float,
        default=10.0,
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
        robot.move_to_joint_pos([0, -math.pi / 4, math.pi / 4, 0.0, 0.0, 0.0])
        time.sleep(0.5)
        robot.switch_mode(RobotMode.SERVO_JOINT_POS)
        start = time.time_ns()
        try:
            while True:
                timed_coef = math.sin(
                    (time.time_ns() - start - 0.5 * 1e9)
                    / 1e9
                    * 2
                    * math.pi
                    / args.duration
                )
                joint_pos = [
                    math.pi / 2 * timed_coef,
                    -math.pi / 4 - math.pi / 4 * timed_coef,
                    math.pi / 4 + math.pi / 4 * timed_coef,
                    0,
                    0,
                    0,
                ]
                robot.servo_joint_pos(joint_pos)
                robot.servo_eef_pos([(1 - timed_coef) * 0.07 / 2])
                time.sleep(0.01)
        finally:
            robot.set_speed_profile(SpeedProfile.DEFAULT)


if __name__ == "__main__":
    task_swing()
