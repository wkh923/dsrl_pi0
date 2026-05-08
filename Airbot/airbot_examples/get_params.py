"""
Example of getting parameters from control service.
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


def get_params():
    """
    Example of getting parameters from control service.
    """
    import argparse

    from airbot_py.arm import AIRBOTPlay

    parser = argparse.ArgumentParser(description="Get params example")
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
        "params",
        action="append",
        default=[],
        help="Parameters to get",
    )

    args = parser.parse_args()

    with AIRBOTPlay(url=args.url, port=args.port) as robot:
        print(robot.get_params(args.params))


if __name__ == "__main__":
    get_params()
