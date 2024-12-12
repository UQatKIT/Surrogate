"""Executable script for surrogate test client.

Execute this script to probe an already running control server.
For info on how to run the script, type `python run_server.py --help` in the command line.

Functions:
    process_cli_arguments: Read in command-line arguments for application to run.
    main: Main routine to be invoked when script is executed
"""

import argparse
import importlib
import time

from surrogate import test_client


# ==================================================================================================
def process_cli_arguments() -> tuple[str, float]:
    """Read in command-line arguments for application to run.

    Every application has to provide a `settings.py` file.

    Returns:
        tuple[str, float]: Application directory, sleep time for data pickling
    """
    arg_parser = argparse.ArgumentParser(
        prog="run_testclient.py",
        usage="python %(prog)s [options]",
        description="Run file for surrogate test client",
    )

    arg_parser.add_argument(
        "-app",
        "--application",
        type=str,
        required=True,
        help="Application directory",
    )

    arg_parser.add_argument(
        "-t",
        "--sleeptime",
        type=float,
        required=False,
        default=1,
        help="Artificial sleep time to allow for data pickling",
    )

    cli_args = arg_parser.parse_args()
    application_dir = cli_args.application.replace("/", ".").strip(".")
    sleep_time = cli_args.sleeptime

    return application_dir, sleep_time


# ==================================================================================================
def main() -> None:
    """Main routine, reads in settings and runs a TestClient."""
    application_dir, sleep_time = process_cli_arguments()
    settings_module = f"{application_dir}.settings"
    settings = importlib.import_module(settings_module)

    client = test_client.TestClient(settings.test_client_settings)
    print("Run test client...")
    client.run()
    time.sleep(sleep_time)


if __name__ == "__main__":
    main()
