import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata


# ==================================================================================================
def visualize_checkpoint(pdf, surrogate, test_params, surrogate_evals=None):
    param_dim = test_params.shape[1]
    if param_dim == 1:
        _visualize_checkpoint_1D(pdf, surrogate, test_params, surrogate_evals)
    elif param_dim == 2:
        _visualize_checkpoint_2D(pdf, surrogate, test_params, surrogate_evals)
    else:
        raise ValueError(f"Unsupported parameter dimension: {param_dim}")


# --------------------------------------------------------------------------------------------------
def _visualize_checkpoint_1D(pdf, surrogate, test_params, surrogate_evals):
    training_data = surrogate.training_data
    input_training = training_data[0]
    output_training = training_data[1]
    mean, std = process_mean_std(surrogate, test_params)

    fig, ax = plt.subplots(layout="constrained")
    ax.plot(test_params, mean)
    ax.scatter(input_training, output_training, marker="x", color="red")
    ax.fill_between(test_params[:, 0], mean - 1.96 * std, mean + 1.96 * std, alpha=0.2)
    if surrogate_evals is not None:
        input_values = [value_pair[0] for value_pair in surrogate_evals]
        output_values = [value_pair[1] for value_pair in surrogate_evals]
        ax.scatter(input_values, output_values, marker="o", color="green")
    pdf.savefig(fig)
    plt.close(fig)


# --------------------------------------------------------------------------------------------------
def _visualize_checkpoint_2D(pdf, surrogate, test_params, surrogate_evals):
    training_data = surrogate.training_data
    input_training = training_data[0]
    mean, std = process_mean_std(surrogate, test_params)
    x_grid = np.linspace(np.min(test_params[:, 0]), np.max(test_params[:, 0]), 100)
    y_grid = np.linspace(np.min(test_params[:, 1]), np.max(test_params[:, 1]), 100)
    x_grid, y_grid = np.meshgrid(x_grid, y_grid)
    interpolated_mean = griddata(test_params, mean, (x_grid, y_grid), method="linear")
    interpolated_variance = griddata(test_params, std, (x_grid, y_grid), method="linear")

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), layout="constrained")
    cplot_mean = ax[0].contourf(x_grid, y_grid, interpolated_mean, levels=50, cmap="Blues")
    cplot_var = ax[1].contourf(x_grid, y_grid, interpolated_variance, levels=50, cmap="Blues")
    ax[0].scatter(input_training[:, 0], input_training[:, 1], marker="x", color="red")
    ax[0].set_title("Mean")
    ax[1].set_title("Standard Deviation")
    fig.colorbar(cplot_mean)
    fig.colorbar(cplot_var)
    if surrogate_evals is not None and len(surrogate_evals) > 0:
        input_values = np.array([value_pair[0] for value_pair in surrogate_evals])
        ax[0].scatter(input_values[:, 0], input_values[:, 1], marker="o", color="green")
    pdf.savefig(fig)
    plt.close(fig)


# --------------------------------------------------------------------------------------------------
def process_mean_std(surrogate, test_params):
    mean, variance = surrogate.predict_and_estimate_variance(test_params)

    if surrogate.variance_is_relative:
        if surrogate.variance_reference is not None:
            reference_variance = surrogate.variance_reference
        else:
            reference_variance = np.maximum(surrogate.output_data_range**2, 1e-6)
        variance /= np.sqrt(reference_variance)

    std = np.sqrt(variance)
    if surrogate.log_transformed:
        mean = np.exp(mean)

    return mean, std
