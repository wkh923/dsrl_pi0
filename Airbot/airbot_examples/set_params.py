"""
Example of setting parameters to control service.
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


def set_params():
    """
    Example of setting parameters to control service.
    """
    import argparse
    import json

    from airbot_py.arm import AIRBOTPlay

    parser = argparse.ArgumentParser(description="Set params example")
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
        default={},
        type=json.loads,
        help="Parameters to get, in json format.",
    )

    args = parser.parse_args()

    with AIRBOTPlay(url=args.url, port=args.port) as robot:
        robot.set_params(args.params)


if __name__ == "__main__":
    set_params()
