"""
Example of switching control mode.
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


def switch_mode():
    """
    Switch control mode according to the given mode
    """

    import argparse

    from airbot_py.arm import AIRBOTPlay, RobotMode

    parser = argparse.ArgumentParser(description="Switch control mode example")
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        choices=[
            "planning_pos",
            "planning_waypoints_pos",
            "servo_joint_pos",
            "servo_joint_vel",
            "servo_cart_pose",
            "servo_cart_twist",
            "gravity_comp",
        ],
        default="planning_pos",
        help="control mode to switch",
    )
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
    args = parser.parse_args()

    with AIRBOTPlay(url=args.url, port=args.port) as robot:
        if args.mode == "planning_pos":
            robot.switch_mode(RobotMode.PLANNING_POS)
            print("Entered planning position mode")
        elif args.mode == "planning_waypoints_pos":
            robot.switch_mode(RobotMode.PLANNING_WAYPOINTS_POS)
            print("Entered planning waypointss mode")
        elif args.mode == "servo_joint_pos":
            robot.switch_mode(RobotMode.SERVO_JOINT_POS)
            print("Entered servo joint position mode")
        elif args.mode == "servo_joint_vel":
            robot.switch_mode(RobotMode.SERVO_JOINT_VEL)
            print("Entered servo joint velocity mode.")
        elif args.mode == "servo_cart_pose":
            robot.switch_mode(RobotMode.SERVO_CART_POSE)
            print("Entered servo cartesian pose mode.")
        elif args.mode == "servo_cart_twist":
            robot.switch_mode(RobotMode.SERVO_CART_TWIST)
            print("Entered servo cartesian twist mode.")
        elif args.mode == "gravity_comp":
            robot.switch_mode(RobotMode.GRAVITY_COMP)
            print("Entered gravity compensation mode.")
            print(
                "\x1b[1;32mGravity compensation mode. You can drag the end of the robot arm freely now.\x1b[0m"
            )
        else:
            print("Invalid mode")


if __name__ == "__main__":
    switch_mode()
