import argparse
import importlib

import src.surrogate.visualization as visualization


# ==================================================================================================
def process_cli_arguments():
    argParser = argparse.ArgumentParser(
        prog="run_visualization.py",
        usage="python %(prog)s [options]",
        description="Run file for visualization of surrogate checkpoints",
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
    settings_module = f"{application_dir}.settings_fit"
    settings = importlib.import_module(settings_module)

    surrogate_settings = settings.surrogate_model_settings
    surrogate_settings.checkpoint_load_file = None
    surrogate_settings.checkpoint_save_path = None

    surrogate_model = settings.surrogate_model_type(settings.surrogate_model_settings)

    # surrogate_model.load_checkpoint(surrogate_settings.checkpoint_load_file)
    # print("Fitting the surrogate model...")
    # surrogate_model.fit()

    visualizer = visualization.Visualizer(settings.visualization_settings, surrogate_model)
    print("Run visualization with training...")
    visualizer.run()


if __name__ == "__main__":
    main()
