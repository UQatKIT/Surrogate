import argparse
import importlib

import umbridge as ub

from surrogate import surrogate_control, utilities


# ==================================================================================================
def process_cli_arguments() -> str:
    arg_parser = argparse.ArgumentParser(
        prog="run_server.py",
        usage="python %(prog)s [options]",
        description="Run file for Umbridge surrogate",
    )

    arg_parser.add_argument(
        "-app",
        "--application",
        type=str,
        required=True,
        help="Application directory",
    )

    cli_args = arg_parser.parse_args()
    application_dir = cli_args.application.replace("/", ".").strip(".")

    return application_dir


# ==================================================================================================
def main() -> None:
    application_dir = process_cli_arguments()
    settings_module = f"{application_dir}.settings"
    settings_module = importlib.import_module(settings_module)

    surrogate_model = settings_module.surrogate_model_type(settings_module.surrogate_model_settings)
    simulation_model = utilities.request_umbridge_server(
        settings_module.simulation_model_settings.url,
        settings_module.simulation_model_settings.name,
    )
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
