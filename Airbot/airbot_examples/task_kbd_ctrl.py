"""
Example application of controlling the robot arm using keyboard input.
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


def task_kbd_ctrl():
    """
    Example application of controlling the robot arm using keyboard input.
    """
    import argparse
    import curses
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

    args = parser.parse_args()

    def main(stdscr):
        stdscr.nodelay(True)
        state = 1
        speed_profile = SpeedProfile.SLOW
        with AIRBOTPlay(url=args.url, port=args.port) as robot:
            robot.switch_mode(RobotMode.SERVO_CART_TWIST)
            robot.set_speed_profile(speed_profile)
            speed = 10
            try:
                while True:
                    key = stdscr.getch()
                    if key != -1:
                        if key == ord("\n"):
                            if speed_profile == SpeedProfile.SLOW:
                                speed_profile = SpeedProfile.DEFAULT
                            elif speed_profile == SpeedProfile.DEFAULT:
                                speed_profile = SpeedProfile.SLOW
                            robot.set_speed_profile(speed_profile)
                        if key == ord(" "):
                            if state == 1:
                                robot.switch_mode(RobotMode.SERVO_JOINT_VEL)
                                state = 0
                            elif state == 0:
                                robot.switch_mode(RobotMode.SERVO_CART_TWIST)
                                state = 1
                        if state == 0:
                            if key == ord("1"):
                                robot.servo_joint_vel([speed, 0.0, 0.0, 0.0, 0.0, 0.0])
                            if key == ord("2"):
                                robot.servo_joint_vel([-speed, 0.0, 0.0, 0.0, 0.0, 0.0])
                            if key == ord("3"):
                                robot.servo_joint_vel([0.0, speed, 0.0, 0.0, 0.0, 0.0])
                            if key == ord("4"):
                                robot.servo_joint_vel([0.0, -speed, 0.0, 0.0, 0.0, 0.0])
                            if key == ord("5"):
                                robot.servo_joint_vel([0.0, 0.0, speed, 0.0, 0.0, 0.0])
                            if key == ord("6"):
                                robot.servo_joint_vel([0.0, 0.0, -speed, 0.0, 0.0, 0.0])
                            if key == ord("7"):
                                robot.servo_joint_vel([0.0, 0.0, 0.0, speed, 0.0, 0.0])
                            if key == ord("8"):
                                robot.servo_joint_vel([0.0, 0.0, 0.0, -speed, 0.0, 0.0])
                            if key == ord("9"):
                                robot.servo_joint_vel([0.0, 0.0, 0.0, 0.0, speed, 0.0])
                            if key == ord("0"):
                                robot.servo_joint_vel([0.0, 0.0, 0.0, 0.0, -speed, 0.0])
                            if key == ord("-"):
                                robot.servo_joint_vel([0.0, 0.0, 0.0, 0.0, 0.0, speed])
                            if key == ord("="):
                                robot.servo_joint_vel([0.0, 0.0, 0.0, 0.0, 0.0, -speed])
                        elif state == 1:
                            if key == ord("a"):
                                robot.servo_cart_twist(
                                    [[0.0, +speed, 0.0], [0.0, 0.0, 0.0]]
                                )
                            if key == ord("d"):
                                robot.servo_cart_twist(
                                    [[0.0, -speed, 0.0], [0.0, 0.0, 0.0]]
                                )
                            if key == ord("w"):
                                robot.servo_cart_twist(
                                    [[+speed, 0.0, 0.0], [0.0, 0.0, 0.0]]
                                )
                            if key == ord("s"):
                                robot.servo_cart_twist(
                                    [[-speed, 0.0, 0.0], [0.0, 0.0, 0.0]]
                                )
                            if key == ord("q"):
                                robot.servo_cart_twist(
                                    [[0.0, 0.0, +speed], [0.0, 0.0, 0.0]]
                                )
                            if key == ord("e"):
                                robot.servo_cart_twist(
                                    [[0.0, 0.0, -speed], [0.0, 0.0, 0.0]]
                                )
                            if key == ord("o"):
                                robot.servo_cart_twist(
                                    [[0.0, 0.0, 0.0], [+speed, 0.0, 0.0]]
                                )
                            if key == ord("u"):
                                robot.servo_cart_twist(
                                    [[0.0, 0.0, 0.0], [-speed, 0.0, 0.0]]
                                )
                            if key == ord("i"):
                                robot.servo_cart_twist(
                                    [[0.0, 0.0, 0.0], [0.0, +speed, 0.0]]
                                )
                            if key == ord("k"):
                                robot.servo_cart_twist(
                                    [[0.0, 0.0, 0.0], [0.0, -speed, 0.0]]
                                )
                            if key == ord("j"):
                                robot.servo_cart_twist(
                                    [[0.0, 0.0, 0.0], [0.0, 0.0, +speed]]
                                )
                            if key == ord("l"):
                                robot.servo_cart_twist(
                                    [[0.0, 0.0, 0.0], [0.0, 0.0, -speed]]
                                )
                        if key == ord("\t"):
                            robot.switch_mode(RobotMode.PLANNING_POS)
                            robot.move_to_joint_pos(
                                [
                                    0.0,
                                    -1.1840663267948965,
                                    1.1840663267948965,
                                    -90 / 360.0 * 2 * math.pi,
                                    90 / 360.0 * 2 * math.pi,
                                    0.0,
                                ]
                            )
                            time.sleep(0.5)
                            robot.switch_mode(RobotMode.SERVO_CART_TWIST)
                            state = 1
                    if key == ord("["):
                        robot.servo_eef_pos([0.00])
                    if key == ord("]"):
                        robot.servo_eef_pos([0.07])
                    if key == ord("z"):  # 按 'q' 退出
                        break
            finally:
                robot.set_speed_profile(SpeedProfile.DEFAULT)

    curses.wrapper(main)


if __name__ == "__main__":
    task_kbd_ctrl()
