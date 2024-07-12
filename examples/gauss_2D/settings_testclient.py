from pathlib import Path

import numpy as np
import src.surrogate.test_client as test_client


# ==================================================================================================
test_client_settings = test_client.TestClientSettings(
    surrogate_url="http://localhost:4243",
    surrogate_name="surrogate",
    simulation_config={"order": 4},
    online_checkpoint_path=Path("results_example_gauss_2D"),
    offline_checkpoint_path=Path("results_example_gauss_2D/surrogate_checkpoint_pretraining.pkl"),
    visualization_file=Path("results_example_gauss_2D/online.pdf"),
    training_params=np.random.uniform(-0.9, 0.9, (5, 2)),
    test_params=np.column_stack(
        (np.repeat(np.linspace(-1, 1, 100), 100), np.tile(np.linspace(-1, 1, 100), 100))
    ),
)
