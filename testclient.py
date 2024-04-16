import argparse
import copy
import importlib
import os
import time
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

import src.surrogate.utilities as utils
    

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

    argParser.add_argument(
        "-pt",
        "--pretraining",
        action="store_true",
        required=False,
        default=False,
        help="Whether to assess pre-training",
    )

    argParser.add_argument(
        "-c",
        "--control",
        action="store_true",
        required=False,
        default=False,
        help="Whether to assess online surrogate",
    )

    cliArgs = argParser.parse_args()
    application_dir = cliArgs.application.replace("/", ".").strip(".")
    assess_pretraining = cliArgs.pretraining
    assess_control = cliArgs.control

    return application_dir, assess_pretraining, assess_control


# --------------------------------------------------------------------------------------------------
def assess_offline_training(pretraining_settings, test_client_settings):
    print("Assess pretraining...")
    pretraining_surrogate = pretraining_settings.surrogate_model_type(
        pretraining_settings.surrogate_model_settings
    )
    pretraining_surrogate.load_checkpoint(test_client_settings.offline_checkpoint_path)

    if not test_client_settings.offline_visualization_file.parent.is_dir():
        test_client_settings.offline_visualization_file.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(test_client_settings.offline_visualization_file) as pdf:
        _visualize_1D(pdf, pretraining_surrogate, test_client_settings.offline_test_params)


# --------------------------------------------------------------------------------------------------
def assess_online_training(control_settings, online_settings):
    print("Assess online control...")

    control = utils.request_umbridge_server(
        online_settings.surrogate_url, online_settings.surrogate_name
    )
    control_call = partial(control, config=online_settings.simulation_config)
    test_surrogate = control_settings.surrogate_model_type(
        control_settings.surrogate_model_settings
    )

    evals_before_each_checkpoint = [[]]
    for param in online_settings.online_training_params:
        result = control_call([[param]])
        value_pair = [param, np.exp(result[0][0])]
        surrogate_used = result[2][0]
        if surrogate_used:
            evals_before_each_checkpoint[-1].append(value_pair)
        else:
            last_evals = copy.deepcopy(evals_before_each_checkpoint[-1])
            evals_before_each_checkpoint.append(last_evals)
    num_checkpoints = len(evals_before_each_checkpoint) - 1
    time.sleep(1)

    checkpoint_files = _find_checkpoints(online_settings.checkpoint_path)
    test_params = online_settings.online_test_params
    with PdfPages(online_settings.online_visualization_file) as pdf:
        test_surrogate.load_checkpoint(online_settings.offline_checkpoint_path)
        _visualize_1D(pdf, test_surrogate, test_params, evals_before_each_checkpoint[0])

        for i in range(num_checkpoints):
            checkpoint_file = [file for file in checkpoint_files if f"{i}" in file][0]
            test_surrogate.load_checkpoint(online_settings.checkpoint_path / checkpoint_file)
            _visualize_1D(pdf, test_surrogate, test_params, evals_before_each_checkpoint[i + 1])


# --------------------------------------------------------------------------------------------------
def _find_checkpoints(directory):
    files = []
    for file in os.listdir(directory):
        if ("surrogate_checkpoint" in file) and ("offline" not in file):
            files.append(file)
    return files


# --------------------------------------------------------------------------------------------------
def _visualize_1D(pdf, surrogate, test_params, surrogate_evals=None):
    training_data = surrogate.training_data
    mean, variance = surrogate.predict_and_estimate_variance(test_params)

    if surrogate.variance_is_relative:
        if surrogate.variance_reference is not None:
            reference_variance = surrogate.variance_reference
        else:
            reference_variance = np.maximum(surrogate.output_data_range**2, 1e-6)
        variance /= np.sqrt(reference_variance)

    std = np.sqrt(variance)
    input_training = training_data[0]
    output_training = training_data[1]
    mean = np.exp(mean)

    fig, ax = plt.subplots()
    ax.plot(test_params, mean)
    ax.scatter(input_training, output_training, marker="x", color="red")
    ax.fill_between(test_params[:, 0], mean - 1.96 * std, mean + 1.96 * std, alpha=0.2)
    if surrogate_evals is not None:
        input_values = [value_pair[0] for value_pair in surrogate_evals]
        output_values = [value_pair[1] for value_pair in surrogate_evals]
        ax.scatter(input_values, output_values, marker="o", color="green")
    pdf.savefig(fig)
    plt.close(fig)


# ==================================================================================================
def main():
    application_dir, assess_pretraining, assess_control = process_cli_arguments()
    settings_testclient = f"{application_dir}.settings_testclient"
    settings_pretraining = f"{application_dir}.settings_pretraining"
    settings_control = f"{application_dir}.settings_control"
    settings_testclient = importlib.import_module(settings_testclient)
    settings_pretraining = importlib.import_module(settings_pretraining)
    settings_control = importlib.import_module(settings_control)

    if assess_pretraining:
        assess_offline_training(settings_pretraining, settings_testclient.pretraining_settings)
    if assess_control:
        assess_online_training(settings_control, settings_testclient.online_settings)


if __name__ == "__main__":
    main()
