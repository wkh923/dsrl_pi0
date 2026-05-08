"""
Example of moving the robot arm with waypoints.
"""


def move_with_cart_waypoints():
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
        "--cart-waypoints",
        type=json.loads,
        help="Target waypoints as an array",
        default=[
            [[0.46, 0.0, 0.36], [0.0, 0.0, 0.0, 1.0]],
            [[0.46, 0.1, 0.26], [0.0, 0.0, 0.0, 1.0]],
            [[0.46, 0.0, 0.16], [0.0, 0.0, 0.0, 1.0]],
            [[0.46, -0.1, 0.26], [0.0, 0.0, 0.0, 1.0]],
        ],
    )
    parser.add_argument(
        "-S",
        "--speed",
        choices=["slow", "fast", "default"],
        default="default",
        help="Speed profile for the robot arm",
    )

    parser.add_argument(
        "--path",
        action="store_true",
        help="Execute path without planning",
    )

    args = parser.parse_args()
    with AIRBOTPlay(url=args.url, port=args.port) as robot:
        if args.speed == "slow":
            robot.set_speed_profile(SpeedProfile.SLOW)
        elif args.speed == "fast":
            robot.set_speed_profile(SpeedProfile.FAST)
        else:
            robot.set_speed_profile(SpeedProfile.DEFAULT)
        if args.path:
            robot.switch_mode(RobotMode.PLANNING_WAYPOINTS_PATH)
        else:
            robot.switch_mode(RobotMode.PLANNING_WAYPOINTS)
        robot.move_with_cart_waypoints(args.cart_waypoints)
        robot.set_speed_profile(SpeedProfile.DEFAULT)


if __name__ == "__main__":
    move_with_cart_waypoints()
