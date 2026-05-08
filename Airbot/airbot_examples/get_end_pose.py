"""
Example of getting end effector pose.

The end effector pose along with robot state and control mode will be printed.
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


def get_end_pose():
    """
    Get end effector pose
    """
    import argparse
    import time

    from airbot_py.arm import AIRBOTPlay

    parser = argparse.ArgumentParser(description="Get arm information example 2")
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

        while True:
            ####################################################
            # Get joint position and end effector position start
            pose = robot.get_end_pose()
            # Get joint position and end effector position end
            ####################################################
            if idx % args.freq == 0:
                server_state = robot.get_state().name
                control_mode = robot.get_control_mode().name
                print("---------------------------------------------------------")
                print(f"Server state: {server_state}")
                print(f"Control mode: {control_mode}")
                print(
                    "".join(
                        [f"{i:>8s}" for i in ["x", "y", "z", "qx", "qy", "qz", "qw"]]
                    )
                )
                print("---------------------------------------------------------")
            print(
                (
                    "".join([f"{i:+8.2f}" for i in pose[0]])
                    if pose is not None
                    else "Not received yet"
                ),
                end="",
            )
            print("".join([f"{i:+8.2f}" for i in pose[1]]) if pose is not None else "")
            time.sleep(1 / args.freq)
            idx += 1


if __name__ == "__main__":
    get_end_pose()
