import argparse
import importlib

import umbridge as ub

import surrogate.surrogate_control as surrogate_control


# ==================================================================================================
def process_cli_arguments():
    argParser = argparse.ArgumentParser(
        prog="surrogate.py",
        usage="python %(prog)s [options]",
        description="Run file for Umbridge surrogate",
    )

    argParser.add_argument(
        "-s",
        "--settings",
        type=str,
        required=False,
        default="settings",
        help="Settings file",
    )

    cliArgs = argParser.parse_args()
    settings_module = cliArgs.settings.replace("/", ".").strip(".")

    return settings_module


# ==================================================================================================
def main():
    settings_module = process_cli_arguments()
    settings_module = importlib.import_module(settings_module)

    print("\n=========== Start Umbridge Surrogate ===========\n")
    model = settings_module.surrogate_model_type(settings_module.surrogate_model_settings)
    control = surrogate_control.SurrogateControl(
        settings_module.surrogate_control_settings,
        model,
        settings_module.simulation_model,
    )
    ub.serve_models(
        [control], port=settings_module.surrogate_control_settings.port, max_workers=100
    )


if __name__ == "__main__":
    main()
