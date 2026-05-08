"""
Example of sending continuous mit joint commands to the robot arm.
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


def mit_joint_integrated():
    """
    Example of sending continuous joint position commands to the robot arm.
    """

    import argparse
    import json
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
        "--pos",
        type=json.loads,
        help="Target joint positions, a 6-float array",
        default=[0.0, -1.19, 1.17, -1.55, 1.51, -0.00],
    )
    parser.add_argument(
        "--vel",
        type=json.loads,
        help="Target joint velocities, a 6-float array",
        default=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    parser.add_argument(
        "--eff",
        type=json.loads,
        help="Target joint efforts, a 6-float array",
        default=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    parser.add_argument(
        "--kp",
        type=json.loads,
        help="Target joint kps, a 6-float array",
        default=[90.0, 150.0, 150.0, 25.0, 25.0, 25.0],
    )
    parser.add_argument(
        "--kd",
        type=json.loads,
        help="Target joint kds, a 6-float array",
        default=[2.5, 1.75, 1.5, 0.5, 1.5, 0.5],
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
            pos = [float(i) for i in args.pos]
            vel = [float(i) for i in args.vel]
            eff = [float(i) for i in args.eff]
            kp = [float(i) for i in args.kp]
            kd = [float(i) for i in args.kd]
            # time.sleep(3)
            robot.move_to_joint_pos(pos)
            robot.switch_mode(RobotMode.MIT_INTEGRATED)
            import time

            joint1_upper = 1.57
            joint1_lower = -1.57
            delta_pos = 0.001 * 1.5
            direction = 1

            while True:
                robot.mit_joint_integrated_control(pos, vel, eff, kp, kd)
                if direction == 1:
                    pos[0] += delta_pos
                    if pos[0] >= joint1_upper:
                        pos[0] = joint1_upper
                        direction = -1
                else:
                    pos[0] -= delta_pos
                    if pos[0] <= joint1_lower:
                        pos[0] = joint1_lower
                        direction = 1
                time.sleep(0.8 / 250)
        finally:
            robot.set_speed_profile(SpeedProfile.DEFAULT)


if __name__ == "__main__":
    mit_joint_integrated()
