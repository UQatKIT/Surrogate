import argparse
import importlib

import src.surrogate.offline_training as offline_training
import src.surrogate.utilities as utils


# ==================================================================================================
def process_cli_arguments():
    argParser = argparse.ArgumentParser(
        prog="run_pretraining.py",
        usage="python %(prog)s [options]",
        description="Run file for surrogate pre-training",
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
    settings = importlib.import_module(settings_module)

    surrogate_settings = settings.surrogate_model_settings
    surrogate_settings.checkpoint_load_file = None
    surrogate_model = settings.surrogate_model_type(surrogate_settings)
    simulation_model = utils.request_umbridge_server(
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
