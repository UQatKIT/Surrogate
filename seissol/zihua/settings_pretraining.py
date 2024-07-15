from pathlib import Path

from sklearn.gaussian_process.kernels import RBF, ConstantKernel

import src.surrogate.offline_training as offline_training
import src.surrogate.surrogate_model as surrogate_model
import src.surrogate.utilities as utils

# ==================================================================================================
simulation_model_settings = utils.SimulationModelSettings(
    url="http://localhost:4343",
    name="QueuingModel",
)

surrogate_model_type = surrogate_model.SKLearnGPSurrogateModel

surrogate_model_settings = surrogate_model.SKLearnGPSettings(
    scaling_kernel=ConstantKernel(constant_value=0.5, constant_value_bounds=(1e-5, 1e5)),
    correlation_kernel=RBF(length_scale=0.1, length_scale_bounds=(1e-2, 1e1)),
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
    checkpoint_load_file=None,
    checkpoint_save_path=Path("results_sigyS_seissol_zihua"),
)

# --------------------------------------------------------------------------------------------------
pretraining_settings = offline_training.OfflineTrainingSettings(
    num_offline_training_points=5,
    num_threads=5,
    offline_model_config={"meshFile": "Ridgecrest_NewModel1_f200_topo1000_noRef_xml_UBC","order": 3},
    lhs_bounds=[[0.5, 3.0]],
    lhs_seed=0,
    checkpoint_save_name="pretraining",
)

pretraining_logger_settings = utils.LoggerSettings(
    do_printing=True,
    logfile_path=Path("results_sigyS_seissol_zihua/pretraining.log"),
    write_mode="w",
)
