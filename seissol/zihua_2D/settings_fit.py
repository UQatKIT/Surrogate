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
    url="http://localhost:4343",
    name="QueuingModel",
)

surrogate_model_type = surrogate_model.SKLearnGPSurrogateModel

surrogate_model_settings = surrogate_model.SKLearnGPSettings(
    scaling_kernel=ConstantKernel(constant_value=0.5, constant_value_bounds=(1e-5, 1e5)),
    correlation_kernel=RBF(length_scale=(1, 1), length_scale_bounds=((1e-5, 1e5),(1e-5, 1e5))),
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
    checkpoint_load_file="../output_surro/results_seissol_zihua_2D_T18/surrogate_checkpoint_2.pkl",
    #checkpoint_load_file=None,
    checkpoint_save_path=Path("results_seissol_zihua_2D_T19"),
)

# --------------------------------------------------------------------------------------------------
surrogate_control_settings = surrogate_control.ControlSettings(
    name="surrogate",
    port=4243,
    minimum_num_training_points=0,
    variance_threshold=1e-3,
    update_interval_rule=lambda num_updates: num_updates + 1,
    overwrite_checkpoint=False,
)

control_logger_settings = utils.LoggerSettings(
    do_printing=True,
    logfile_path=Path("results_seissol_zihua_2D_T19/online.log"),
    write_mode="w",
)

# --------------------------------------------------------------------------------------------------
pretraining_settings = offline_training.OfflineTrainingSettings(
    num_offline_training_points=10,
    num_threads=5,
    offline_model_config={"meshFile": "Ridgecrest_NewModel1_f200_topo1000_noRef_xml_UBC","order": 3},
    lhs_bounds=[[0.5, 2.5], [0.3, 0.9]],
    lhs_seed=0,
    checkpoint_save_name="pretraining",
)

pretraining_logger_settings = utils.LoggerSettings(
    do_printing=True,
    logfile_path=Path("results_seissol_zihua_2D_T19/pretraining.log"),
    write_mode="w",
)

# --------------------------------------------------------------------------------------------------
test_client_settings = test_client.TestClientSettings(
    control_url="http://localhost:4243",
    control_name="surrogate",
    simulation_config={"meshFile": "Ridgecrest_NewModel1_f200_topo1000_noRef_xml_UBC","order": 3},
    training_params=np.random.uniform(0.5, 0.9, (5, 2)),
)

# --------------------------------------------------------------------------------------------------
visualization_settings = visualization.VisualizationSettings(
    offline_checkpoint_file=Path("../output_surro/results_seissol_zihua_2D_T18/surrogate_checkpoint_2.pkl"),
    online_checkpoint_filestub=Path("../output_surro/results_seissol_zihua_2D_T19/surrogate_checkpoint"),
    visualization_file=Path("results_seissol_zihua_2D_T19/visualization.pdf"),
    visualization_points=np.column_stack(
        (np.repeat(np.linspace(0.5, 2.5, 100), 100), np.tile(np.linspace(0.3, 0.9, 100), 100))
    ),
)
