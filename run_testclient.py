import argparse
import importlib
import time

import src.surrogate.test_client as test_client


# ==================================================================================================
def process_cli_arguments():
    argParser = argparse.ArgumentParser(
        prog="test_client.py",
        usage="python %(prog)s [options]",
        description="Run file for surrogate test client",
    )

    argParser.add_argument(
        "-app",
        "--application",
        type=str,
        required=True,
        help="Application directory",
    )

    cliArgs = argParser.parse_args()
    application_dir = cliArgs.application.replace("/", ".").strip(".")

    return application_dir


# ==================================================================================================
def main():
    application_dir = process_cli_arguments()
    settings_testclient = f"{application_dir}.settings_testclient"
    settings_control = f"{application_dir}.settings_control"
    settings_testclient = importlib.import_module(settings_testclient)
    settings_control = importlib.import_module(settings_control)

    test_surrogate_model = settings_control.surrogate_model_type(
        settings_control.surrogate_model_settings
    )
    client = test_client.TestClient(settings_testclient.test_client_settings, test_surrogate_model)
    print("Run test client...")
    client.run()
    time.sleep(1)
    print("Visualize results...")
    client.visualize()


if __name__ == "__main__":
    main()
