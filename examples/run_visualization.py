"""Executable script for surrogate output visualization.

For info on how to run the script, type `python run_server.py --help` in the command line.

Functions:
    process_cli_arguments: Read in command-line arguments for application to run.
    main: Main routine to be invoked when script is executed
"""

import argparse
import importlib

from surrogate import visualization


# ==================================================================================================
def process_cli_arguments() -> str:
    """Read in command-line arguments for application to run.

    Every application has to provide a `settings.py` file.

    Returns:
        str: Application directory
    """
    arg_parser = argparse.ArgumentParser(
        prog="run_visualization.py",
        usage="python %(prog)s [options]",
        description="Run file for visualization of surrogate checkpoints",
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
    """Main routine, reads in settings, sets up a surrogate and runs Visualizer."""
    application_dir = process_cli_arguments()
    settings_module = f"{application_dir}.settings"
    settings = importlib.import_module(settings_module)

    surrogate_settings = settings.surrogate_model_settings
    surrogate_settings.checkpoint_load_file = None
    surrogate_settings.checkpoint_save_path = None

    surrogate_model = settings.surrogate_model_type(settings.surrogate_model_settings)
    visualizer = visualization.Visualizer(settings.visualization_settings, surrogate_model)
    print("Run visualization...")
    visualizer.run()


if __name__ == "__main__":
    main()
