"""
Example of moving the robot arm with waypoints.
"""


def move_with_joint_waypoints():
    """
    Example of moving the robot arm with waypoints..
    """
    import argparse
    import json

    print(
        "\x1b[1;33mThe robot arm is about to be move. Please make sure no obstacles exist in the surrounding environment.\x1b[0m"
    )
    input("\x1b[1;35mPress Enter to continue...\x1b[0m")

    from airbot_py.arm import AIRBOTPlay, RobotMode, SpeedProfile

    parser = argparse.ArgumentParser(description="Waypoints example")
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
        "-w",
        "--joint-waypoints",
        type=json.loads,
        help="Target translation in the cartesian space, a 3-float array",
        default=[
            [0.37, -0.6, 0.5, 0.132, -0.30, -0.13],
            [0.2, -2.0, 2.0, 0.0, 0.01, 0.0],
            [0.2, 0.1, 0.3, 0.0, 0.01, 0.0],
            [-0.2, 0.15, 0.0, 0.0, 0.01, 0.0],
        ],
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
        robot.switch_mode(RobotMode.PLANNING_WAYPOINTS)
        robot.move_with_joint_waypoints(args.joint_waypoints)
        robot.set_speed_profile(SpeedProfile.DEFAULT)


if __name__ == "__main__":
    move_with_joint_waypoints()
