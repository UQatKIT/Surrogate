import argparse
import importlib

from surrogate import offline_training, utilities


# ==================================================================================================
def process_cli_arguments() -> str:
    arg_parser = argparse.ArgumentParser(
        prog="run_pretraining.py",
        usage="python %(prog)s [options]",
        description="Run file for surrogate pre-training",
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
    settings = importlib.import_module(settings_module)

    surrogate_settings = settings.surrogate_model_settings
    surrogate_settings.checkpoint_load_file = None
    surrogate_model = settings.surrogate_model_type(surrogate_settings)
    simulation_model = utilities.request_umbridge_server(
        settings.simulation_model_settings.url,
        settings.simulation_model_settings.name,
    )

    offline_trainer = offline_training.OfflineTrainer(
        settings.pretraining_settings,
        settings.pretraining_logger_settings,
        surrogate_model,
        simulation_model,
    )
    print("Run pretraining...")
    offline_trainer.run()


if __name__ == "__main__":
    main()
