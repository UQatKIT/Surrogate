from pathlib import Path

import numpy as np
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

import src.surrogate.surrogate_control as surrogate_control
import src.surrogate.surrogate_model as surrogate_model
import src.surrogate.offline_training as offline_training
import src.surrogate.test_client as test_client
import src.surrogate.visualization as visualization
import src.surrogate.utilities as utils

# ==================================================================================================
simulation_model_settings = utils.SimulationModelSettings(
    url="http://localhost:4242",
    name="forward",
)

surrogate_model_type = surrogate_model.SKLearnGPSurrogateModel

surrogate_model_settings = surrogate_model.SKLearnGPSettings(
    scaling_kernel=ConstantKernel(constant_value=0.5, constant_value_bounds=(1e-5, 1e5)),
    correlation_kernel=RBF(length_scale=6 * (1,), length_scale_bounds=6 * ((1e-5, 1e5),)),
    data_noise=1e-6,
    num_optimizer_restarts=3,
    minimum_num_training_points=3,
    normalize_output=False,
    perform_log_transform=True,
    variance_is_relative=False,
    variance_reference=None,
    value_range_underflow_threshold=1e-6,
    log_mean_underflow_value=-1000,
    mean_underflow_value=1e-6,
    init_seed=0,
    checkpoint_load_file="results_example_gauss_6D/surrogate_checkpoint_pretraining.pkl",
    checkpoint_save_path=Path("results_example_gauss_6D"),
)

# --------------------------------------------------------------------------------------------------
surrogate_control_settings = surrogate_control.ControlSettings(
    name="surrogate",
    port=4243,
    minimum_num_training_points=0,
    update_interval_rule=lambda num_updates: num_updates + 1,
    variance_threshold=1e-9,
    overwrite_checkpoint=False,
)

control_logger_settings = utils.LoggerSettings(
    do_printing=True,
    logfile_path=Path("results_example_gauss_6D/online.log"),
    write_mode="w",
)

# --------------------------------------------------------------------------------------------------
pretraining_settings = offline_training.OfflineTrainingSettings(
    num_offline_training_points=200,
    num_threads=1,
    offline_model_config={},
    lhs_bounds=6
    * [
        [-1, 1],
    ],
    lhs_seed=0,
    checkpoint_save_name="pretraining",
)

pretraining_logger_settings = utils.LoggerSettings(
    do_printing=True,
    logfile_path=Path("results_example_gauss_6D/pretraining.log"),
    write_mode="w",
)

# --------------------------------------------------------------------------------------------------
test_client_settings = test_client.TestClientSettings(
    control_url="http://localhost:4243",
    control_name="surrogate",
    simulation_config={},
    training_params=np.random.uniform(-0.9, 0.9, (5, 6)),
)

# --------------------------------------------------------------------------------------------------
visualization_settings = visualization.VisualizationSettings(
    offline_checkpoint_file=Path("results_example_gauss_6D/surrogate_checkpoint_pretraining.pkl"),
    online_checkpoint_filestub=Path("results_example_gauss_6D/surrogate_checkpoint"),
    visualization_file=Path("results_example_gauss_6D/visualization.pdf"),
    visualization_bounds=6 * [(-1, 1),]
)
