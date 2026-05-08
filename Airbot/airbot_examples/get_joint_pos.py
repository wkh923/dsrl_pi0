"""
Example of getting joint position.

The joint position along with robot state and control mode will be printed.
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


def get_joint_pos():
    """
    Get joint position
    """
    import argparse
    import math
    import time

    from airbot_py.arm import AIRBOTPlay

    parser = argparse.ArgumentParser(description="Get arm information example 1")
    parser.add_argument(
        "-f",
        "--freq",
        type=int,
        default=10,
        help="Frequency of printing joint positions",
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
    parser.add_argument(
        "--radian",
        action="store_true",
        help="Print joint positions in radians instead of degrees",
    )
    args = parser.parse_args()

    idx = 0
    with AIRBOTPlay(url=args.url, port=args.port) as robot:
        product_info = robot.get_product_info()
        print("---------------------------------------------------------")
        print(f"Product name: {product_info['product_type']}")
        print(f"Serial number: {product_info['sn']}")
        print(f"Simulation mode: {product_info['is_sim']}")
        print(f"Using interfaces: {product_info['interfaces']}")
        print(f"Installed end effectors: {product_info['eef_types']}")
        print(f"Firmware versions: {product_info['fw_versions']}")
        print("---------------------------------------------------------")

        while True:
            ####################################################
            # Get joint position and end effector position start
            pos = robot.get_joint_pos()
            eef_pos = robot.get_eef_pos()
            # Get joint position and end effector position end
            ####################################################
            if idx % args.freq == 0:
                server_state = robot.get_state().name
                control_mode = robot.get_control_mode().name
                print(
                    "----------------------------------------------------------------"
                )
                print(f"Server state: {server_state}")
                print(f"Control mode: {control_mode}")
                print(
                    " ".join([f"{'joint':>7s}{i+1}" for i in range(6)])
                    + f"{'end eff':>10s}"
                )
                print(
                    "----------------------------------------------------------------"
                )
            print(
                (
                    ", ".join(
                        [
                            (
                                f"{i / math.pi * 180:+7.2f}°"
                                if not args.radian
                                else f"{i:7.2f}"
                            )
                            for i in pos
                        ]
                    )
                    if pos is not None
                    else "None received yet"
                ),
                end="",
            )
            print(
                f"{eef_pos[0] * 100:8.2f} cm"
                if eef_pos is not None and len(eef_pos) > 0
                else ""
            )
            time.sleep(1 / args.freq)
            idx += 1


if __name__ == "__main__":
    get_joint_pos()
