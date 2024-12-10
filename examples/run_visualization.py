import argparse
import importlib

from surrogate import visualization


# ==================================================================================================
def process_cli_arguments() -> str:
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
