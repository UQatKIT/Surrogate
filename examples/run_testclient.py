import argparse
import importlib
import time

from surrogate import test_client


# ==================================================================================================
def process_cli_arguments() -> str:
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
    application_dir, sleep_time = process_cli_arguments()
    settings_module = f"{application_dir}.settings"
    settings = importlib.import_module(settings_module)

    client = test_client.TestClient(settings.test_client_settings)
    print("Run test client...")
    client.run()
    time.sleep(sleep_time)


if __name__ == "__main__":
    main()
