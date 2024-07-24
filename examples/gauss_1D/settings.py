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
    correlation_kernel=RBF(length_scale=1e6, length_scale_bounds=(1e5, 1e8)),
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
    checkpoint_load_file="results_example_gauss_1D/surrogate_checkpoint_pretraining.pkl",
    checkpoint_save_path=Path("results_example_gauss_1D"),
)

# --------------------------------------------------------------------------------------------------
surrogate_control_settings = surrogate_control.ControlSettings(
    name="surrogate",
    port=4243,
    minimum_num_training_points=0,
    variance_threshold=1e-3,
    update_interval_rule=lambda num_updates: num_updates + 1,
    checkpoint_load_file=None,
    checkpoint_save_path=Path("results_example_gauss_1D"),
    overwrite_checkpoint=False,
)

control_logger_settings = utils.LoggerSettings(
    do_printing=True,
    logfile_path=Path("results_example_gauss_1D/online.log"),
    write_mode="w",
)

# --------------------------------------------------------------------------------------------------
pretraining_settings = offline_training.OfflineTrainingSettings(
    num_offline_training_points=5,
    num_threads=5,
    offline_model_config={"order": 4},
    lhs_bounds=[[0, 1e7]],
    lhs_seed=0,
    checkpoint_save_name="pretraining",
)

pretraining_logger_settings = utils.LoggerSettings(
    do_printing=True,
    logfile_path=Path("results_example_gauss_1D/pretraining.log"),
    write_mode="w",
)

# --------------------------------------------------------------------------------------------------
test_client_settings = test_client.TestClientSettings(
    control_url="http://localhost:4243",
    control_name="surrogate",
    simulation_config={"order": 4},
    training_params=np.random.uniform(0, 1e7, 10),
)

# --------------------------------------------------------------------------------------------------
visualization_settings = visualization.VisualizationSettings(
    online_checkpoint_path=Path("results_example_gauss_1D"),
    offline_checkpoint_file=Path("results_example_gauss_1D/surrogate_checkpoint_pretraining.pkl"),
    visualization_file=Path("results_example_gauss_1D/visualization.pdf"),
    visualization_points=np.linspace(0, 1e7, 100).reshape(-1, 1),   
)
