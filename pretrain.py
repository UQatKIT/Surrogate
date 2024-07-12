import argparse
import importlib

import src.surrogate.offline_training as offline_training
import src.surrogate.utilities as utils


# ==================================================================================================
def process_cli_arguments():
    argParser = argparse.ArgumentParser(
        prog="pretrain.py",
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
    settings_module = f"{application_dir}.settings_pretraining"
    settings_module = importlib.import_module(settings_module)

    surrogate_model = settings_module.surrogate_model_type(settings_module.surrogate_model_settings)
    simulation_model = utils.request_umbridge_server(
        settings_module.simulation_model_settings.url,
        settings_module.simulation_model_settings.name,
    )

    offline_trainer = offline_training.OfflineTrainer(
        settings_module.pretraining_settings,
        settings_module.pretraining_logger_settings,
        surrogate_model,
        simulation_model,
    )
    print("Run pretraining...")
    offline_trainer.run()
    print("Visualize results...")
    offline_trainer.visualize()


if __name__ == "__main__":
    main()
