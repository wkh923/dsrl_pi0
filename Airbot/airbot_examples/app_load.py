"""
Example of loading an internal app
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


def app_load():
    """
    Load app according to the given app name
    """

    import argparse

    from airbot_py.arm import AIRBOTPlay

    parser = argparse.ArgumentParser(description="Load app example")
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
        "-a",
        "--app_name",
        type=str,
        default="record_replay_app/record_replay_app::RecordReplayApp",
        choices=["record_replay_app/record_replay_app::RecordReplayApp"],
        help="Name of the app to load",
    )

    args = parser.parse_args()
    with AIRBOTPlay(url=args.url, port=args.port) as robot:
        robot.load_app("record_replay_app/record_replay_app::RecordReplayApp")


if __name__ == "__main__":
    app_load()
