import argparse
import importlib

import umbridge as ub

import src.surrogate.surrogate_control as surrogate_control
import src.surrogate.utilities as utils

import sys
# ==================================================================================================
def process_cli_arguments():
    argParser = argparse.ArgumentParser(
        prog="run_server.py",
        usage="python %(prog)s [options]",
        description="Run file for Umbridge surrogate",
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
    settings_module = f"{application_dir}.settings"
    settings_module = importlib.import_module(settings_module)

    surrogate_model = settings_module.surrogate_model_type(settings_module.surrogate_model_settings)
    simulation_model = utils.request_umbridge_server(
        settings_module.simulation_model_settings.url,
        settings_module.simulation_model_settings.name,
    )
    print(f"getting {settings_module.simulation_model_settings.name}")
    print("input size of the simulation model",simulation_model.get_input_sizes())
    sys.stdout.flush()
    control = surrogate_control.SurrogateControl(
        settings_module.surrogate_control_settings,
        settings_module.control_logger_settings,
        surrogate_model,
        simulation_model,
    )
    print("Run surrogate control...")
    ub.serve_models(
        [control], port=settings_module.surrogate_control_settings.port, max_workers=100
    )


if __name__ == "__main__":
    main()
