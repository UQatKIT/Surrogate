from pathlib import Path

import umbridge as ub
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

import surrogate.surrogate_model as surrogate_model
import surrogate.surrogate_control as surrogate_control


# ==================================================================================================
simulation_model = ub.HTTPModel(url="http://localhost:4244", name="simulation_model")

surrogate_model_type = surrogate_model.SKLearnGPSurrogateModel

surrogate_model_settings = surrogate_model.SKLearnGPModelSettings(
    scaling_kernel=ConstantKernel(),
    correlation_kernel=RBF(),
    data_noise=1e-12,
    num_optimizer_restarts=9,
    normalize_output=False,
    perform_log_transform=False,
    init_seed=0,
)

surrogate_control_settings = surrogate_control.SurrogateControlSettings(
    name="surrogate",
    port=4243,
    minimum_num_training_points=3,
    variance_threshold=0.1,
    variance_is_relative=True,
    update_interval_rule=lambda num_updates: num_updates,
    update_cycle_delay=0.1,
    checkpoint_interval=1,
    checkpoint_load_path=None,
    checkpoint_save_path=Path("test_results/surrogate_checkpoint.pkl"),
)
