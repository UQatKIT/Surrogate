import pickle

import numpy as np
import umbridge as ub
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel


url = "http://localhost:4243"
name = "surrogate"
config = {"order": 4}
checkpoint_path = Path("checkpoints")
figure_path = Path("figures.pdf")

surrogate = ub.HTTPModel(url=url, name=name)
surrogate_call = partial(surrogate, config=config)

test_gp = GaussianProcessRegressor(
    kernel=ConstantKernel() * RBF(),
    alpha=1e-6,
    optimizer="fmin_l_bfgs_b",
    n_restarts_optimizer=9,
    normalize_y=True,
    random_state=0,
)

test_params = np.linspace(-1e8, 1e8, 100).reshape(-1, 1)
with open("checkpoints/surrogate_checkpoint_offline.pkl", "rb") as cp_file:
    checkpoint = pickle.load(cp_file)

test_gp.kernel.set_params(**checkpoint.hyperparameters)
test_gp.fit(checkpoint.input_data, checkpoint.output_data)
mean, std = test_gp.predict(test_params, return_std=True)
fig, ax = plt.subplots()
ax.plot(test_params, mean)
ax.fill_between(
    test_params[:, 0],
    mean - 1.96 * std,
    mean + 1.96 * std,
    alpha=0.2,
)
fig.savefig("offline")
plt.close(fig)

input_params = np.linspace(-1e8, 1e8, 50)
output = np.zeros(input_params.shape)

for i, param in enumerate(input_params):
    output[i] = surrogate_call([[param]])[0][0]

pdf = PdfPages(figure_path)

for i in range(0, 48):
    checkpoint_file = checkpoint_path / Path(f"surrogate_checkpoint_{i}.pkl")
    with open(checkpoint_file, "rb") as cp_file:
        checkpoint = pickle.load(cp_file)
        test_gp.kernel.set_params(**checkpoint.hyperparameters)
        test_gp.fit(checkpoint.input_data, checkpoint.output_data)
        mean, var = test_gp.predict(test_params, return_std=True)
        fig, ax = plt.subplots()
        ax.plot(test_params, mean)
        ax.fill_between(
            test_params[:, 0],
            mean - 1.96 * var,
            mean + 1.96 * var,
            alpha=0.2,
        )
        pdf.savefig(fig)
        plt.close(fig)

pdf.close()
