import json
import os
import threading
import time
from queue import Queue

import botorch
import gpytorch
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umbridge
from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize

import sys

class Surrogate(umbridge.Model):

    def __init__(self):
        super().__init__("surrogate")

        # Load the configuration file
        with open("custom_surrogate.json", "r") as f:
            config = json.load(f)

        # Connect to the UM-Bridge model
        self.umbridge_model = umbridge.HTTPModel(config["model_port"], config["model_name"])

        # Get the input and output size from the UM-Bridge Model
        self.input_size = self.umbridge_model.get_input_sizes(None)
        self.output_size = self.umbridge_model.get_output_sizes(None)
        print(self.input_size)
        sys.stdout.flush()

        ## True => no fitting / False => fitting
        self.custom_hyperparameters = config["custom_hyperparameters"]
        # If the hyperparameters are manually set
        if self.custom_hyperparameters:
            # Load the hyperparameters from the configuration file
            self.custom_lengthscale = config["lengthscale"]
            self.custom_outputscale = config["outputscale"]
            self.custom_mean = config["mean"]

        # Handel first three requests
        self.pg_ready = threading.Event()

        # Strat from the checkpoint if existent
        if os.path.exists("checkpoint.pth"):
            # Load the checkpoint data
            loaded_checkpoint = torch.load("checkpoint.pth")
            # Put training data into lists
            self.in_list = loaded_checkpoint["input"]
            self.out_list = loaded_checkpoint["output"]
            # Initialize the Gaussian Process
            self.gp = SingleTaskGP(
                self.in_list,
                self.out_list,
                1e-6 * torch.ones_like(self.out_list),
                outcome_transform=Standardize(self.out_list.shape[1]),
                input_transform=Normalize(self.in_list.shape[1]),
            )

            if not self.custom_hyperparameters:
                # Load the hyperparameters from the checkpoint
                self.custom_lengthscale = loaded_checkpoint["lengthscale"]
                self.custom_outputscale = loaded_checkpoint["outputscale"]
                self.custom_mean = loaded_checkpoint["mean"]

                self.next_fit = loaded_checkpoint["next_fit"]

            # Set the hyperparameters for the gp
            self.gp.covar_module.base_kernel.lengthscale = self.custom_lengthscale
            self.gp.covar_module.outputscale = self.custom_outputscale
            if self.custom_hyperparameters:
                self.gp.mean_module.constant = self.custom_mean
            else:
                self.gp.mean_module.constant.data = self.custom_mean

            # Checkpointing
            # The current number of training data
            self.old_save_size = len(self.out_list)
            # The total time spend on model calculations so far
            self.total_time = loaded_checkpoint["total_time"]

            # Handel first three requests
            self.pg_ready.set()

        # Start from scratch if no checkpoint exists
        else:
            # The Gaussian Process
            self.gp = None
            # Empty lists for the training data
            self.in_list = torch.empty((0, sum(self.input_size)), dtype=torch.double)
            self.out_list = torch.empty((0, sum(self.output_size)), dtype=torch.double)

            # Fitting
            if not self.custom_hyperparameters:
                # The next fitting is performed at #next_fit training data
                self.next_fit = 1

            # Checkpointing
            # The current number of training data
            self.old_save_size = 1
            # The total time spend on model calculations so far
            self.total_time = 0

            # Hande first three requests
            self.zaeler = 1
            self.zaeler_lock = threading.Lock()

        # Manage observations
        # Observations that have not yet been included in training
        self.in_queue = Queue()
        self.out_queue = Queue()
        # Lock for the observation queues
        self.lock = threading.Lock()
        # Signals that new observations are available
        self.update_data = threading.Event()

        # Lock for the posterior calculation
        self.pos_lock = threading.Lock()
        # Threshold for the variance
        self.threshold = config["threshold"]

        # Fitting
        # Number of fittings so far
        self.its = 1
        # To calculate the ratio of model calls to surrogate calls
        # Number of surrogate calls since the last fit
        self.count_scalls = 1
        # Number of model calls since the last fit
        self.count_mcalls = 1
        self.count_lock = threading.Lock()

        # Checkpointing dependends on model calculation time and checkpoint saving time
        # Time to save one observation
        self.single_check = (1e-5 / 2) * (self.in_list.size()[1] + self.out_list.size()[1])
        # Number of unsaved observation data
        self.not_saved_data = 0

        # If a varaince plot is required
        self.plot_enabled = config["plot"]
        if self.plot_enabled:
            # Setting lower and upper bounds for the plot
            self.lower_bound = [config["lower_bound_x"], config["lower_bound_y"]]
            self.upper_bound = [config["upper_bound_x"], config["upper_bound_y"]]
            # Overall number of plots
            self.init = 0
            self.frames = []

        # Incooperate previously made observation data
        if os.path.exists("data.txt") and self.gp is None:
            with open("data.txt", "r") as file:
                for line in file:
                    numbers = line.strip().split(",")
                    numbers = [float(num) for num in numbers]
                    in_data = numbers[: sum(self.input_size)]
                    out_data = numbers[sum(self.input_size) :]
                    self.in_queue.put(torch.tensor([in_data], dtype=torch.double))
                    self.out_queue.put(torch.tensor([out_data], dtype=torch.double))

                self.train_gp(None)
                self.pg_ready.set()
                self.save_checkpoint()

    def get_input_sizes(self, config):

        return self.input_size

    def get_output_sizes(self, config):

        return self.output_size

    def heatmap(self, config, points):
        """Plot the variance from the lower bound the the upper bounds.
        The plot is saved as a .png.
        """

        a = np.linspace(self.lower_bound[0], self.upper_bound[0], 100)
        b = np.linspace(self.lower_bound[1], self.upper_bound[1], 100)

        A, B = np.meshgrid(a, b)
        grid_points = np.column_stack((A.ravel(), B.ravel()))
        grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32)

        # Let the gp calculate the variance for the entire meshgrid
        with self.pos_lock:
            with torch.no_grad():
                predictions = self.gp.posterior(grid_points_tensor)
                variance = predictions.variance
                mean = predictions.mean

        mean = mean.reshape(A.shape)
        variance = variance.reshape(B.shape)

        plt.figure()

        # Adjust the colormap
        plt.contourf(A, B, variance, levels=20, cmap="viridis", vmin=0, vmax=20)

        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]

        # Plot the data points with red 'x' markers
        plt.scatter(x_values, y_values, c="red", marker="x")

        plt.colorbar()
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        # plt.title('GP Variance Visualization')
        plt.title("GP Mean Visualization")
        plt.savefig(f"gpvar_{self.init}.png")
        self.frames.append(f"gpvar_{self.init}.png")

    def fit_gp(self, config, new_gp):

        # Perform fitting if required
        if (
            not self.custom_hyperparameters
            and self.count_mcalls / self.count_scalls >= 0.01
            and (self.gp is None or self.next_fit <= self.out_list.size()[0])
        ):
            # Reset the surrogate calls to model calls ratio variables
            with self.count_lock:
                self.count_scalls = 1
                self.count_mcalls = 1

            # Increase the total number of fittings
            self.its = self.its + 1
            # In case that the number of observations are already larger than the next step, skip to the next required step
            while self.its**3 <= self.out_list.size()[0]:
                self.its = self.its + 1
            self.next_fit = self.its**3

            # Perform the fitting
            with self.pos_lock:
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(new_gp.likelihood, new_gp)
                fit_gpytorch_mll(mll)

            self.custom_lengthscale = new_gp.covar_module.base_kernel.lengthscale
            self.custom_outputscale = new_gp.covar_module.outputscale
            self.custom_mean = new_gp.mean_module.constant

        else:
            # Set old hyperparameters if fitting is not required
            new_gp.covar_module.base_kernel.lengthscale = self.custom_lengthscale
            new_gp.covar_module.outputscale = self.custom_outputscale
            if self.custom_hyperparameters:
                new_gp.mean_module.constant = self.custom_mean
            else:
                new_gp.mean_module.constant.data = self.custom_mean

        return new_gp

    # Reinitialize the gp
    def train_gp(self, config):
        """Reinitializ self.gp on all available observation data."""

        # Append the new observation data to the old trainig data
        with self.lock:
            # Signel that input and output queues are both empty
            self.update_data.clear()
            if self.plot_enabled:
                points = [tensor.squeeze().tolist() for tensor in self.in_queue.queue]
            self.in_list = torch.cat([self.in_list] + list(self.in_queue.queue), dim=0)
            self.in_queue = Queue()
            self.out_list = torch.cat([self.out_list] + list(self.out_queue.queue), dim=0)
            self.out_queue = Queue()

        # Set up a new Gaussian Process
        new_gp = SingleTaskGP(
            self.in_list,
            self.out_list,
            1e-6 * torch.ones_like(self.out_list),
            outcome_transform=Standardize(self.out_list.shape[1]),
            input_transform=Normalize(self.in_list.shape[1]),
        )

        # Add the Hyperparameters
        new_gp = self.fit_gp(config, new_gp)

        # Make the first (expensive) posterior calculation
        infoin = torch.vstack([self.in_list[-1].flatten()], out=None)
        with self.pos_lock:
            with torch.no_grad():
                posterior_new = new_gp.posterior(infoin)

        # Hand over the newly initialized Guassian Process to the gp
        self.gp = new_gp
        # Put the gp into evaluation mode
        self.gp.eval()

        # Plot a heatmap of the variance
        if self.plot_enabled:
            self.init = self.init + 1
            self.heatmap(config, points)

    def generate_new_data(self, parameters, config):
        """Return a new observation."""

        # Track the models calcumation time
        start_time = time.time()

        # Let the UM-Bridge model calculate the output
        model_output = self.umbridge_model(parameters)

        # Calculate the time the calculation took
        elapsed_time = time.time() - start_time

        with self.lock:
            # Put observation into its respective queue
            self.in_queue.put(
                torch.tensor(
                    [[item for sublist in parameters for item in sublist]], dtype=torch.double
                )
            )
            self.out_queue.put(
                torch.tensor(
                    [[item for sublist in model_output for item in sublist]], dtype=torch.double
                )
            )

            # Calculate the average time the UM-Bridge model needs for one computation
            self.total_time += elapsed_time
            self.average_time = self.total_time / (len(self.out_list) + self.out_queue.qsize())

            # Signal that new observation data is available
            self.update_data.set()
        return model_output

    def __call__(self, parameters, config):

        # Befor the first training
        if self.gp is None:
            with self.zaeler_lock:
                if self.zaeler <= 3:
                    self.zaeler += 1
                    # Let the UM-Bridge model calculate the output
                    model_output = self.generate_new_data(parameters, config)
                    return model_output

        # Wait for gp to be initialized
        self.pg_ready.wait()
        # Let the gp calculate the posterior for the given parameters
        infoin = torch.tensor([[item for sublist in parameters for item in sublist]])
        with self.pos_lock:
            with torch.no_grad():
                posterior_ = self.gp.posterior(infoin)

        # Get the gps uncertainty about its prediction
        with torch.no_grad():
            pos_variance = posterior_.variance

        # Check if uncertainty is to high
        if self.threshold < torch.max(pos_variance):

            # Increase the surrogate calls to model calls ratio values
            with self.count_lock:
                self.count_scalls = self.count_scalls + 1
                self.count_mcalls = self.count_mcalls + 1

            # Let the UM-Bridge model calculate the output
            model_output = self.generate_new_data(parameters, config)

            return model_output

        # Increase the surrogate calls to model calls ratio value
        with self.count_lock:
            self.count_scalls = self.count_scalls + 1

        # Let the gp calculate the predictive mean
        with torch.no_grad():
            pos_mean = posterior_.mean
        mean = pos_mean.flatten().tolist()

        # Create a list of the correct size and put in the output values
        gp_output = [
            [mean[count] for _ in range(size)] for count, size in enumerate(self.output_size)
        ]
        # Return the mean
        return gp_output

    def supports_evaluate(self):

        return True

    def save_checkpoint(self):
        """Save observations to checkpoint file."""

        # Number of observations being saved right now
        self.old_save_size = len(self.out_list)

        # If the hyperparameters are manually set they do not need to be saved
        if self.custom_hyperparameters:
            torch.save(
                {"input": self.in_list, "output": self.out_list, "total_time": self.total_time},
                "checkpoint.pth",
            )
        # If the hyperparameters are fitted they need to be saved
        else:
            torch.save(
                {
                    "input": self.in_list,
                    "output": self.out_list,
                    "total_time": self.total_time,
                    "next_fit": self.next_fit,
                    "lengthscale": self.custom_lengthscale,
                    "outputscale": self.custom_outputscale,
                    "mean": self.custom_mean,
                },
                "checkpoint.pth",
            )

    def update_gp_thread(self):
        """Thread responsible to update the gp whenever new observation data is available."""

        while True:
            # Wait until new observation data is available
            self.update_data.wait()

            # If gp has not been initialized yet at least 3 observations need to be available
            if self.gp is None:
                if not self.out_queue.qsize() < 3:
                    self.train_gp(None)
                    self.pg_ready.set()
                    self.save_checkpoint()

            else:
                self.train_gp(None)

                # Check if saving a checkpoint right now is more expensive than recalculation the lost data
                if self.single_check * len(self.out_list) < self.average_time * (
                    len(self.out_list) - self.old_save_size
                ):
                    self.save_checkpoint()


testmodel = Surrogate()

update_thread = threading.Thread(target=testmodel.update_gp_thread, daemon=True)
update_thread.start()

umbridge.serve_models([testmodel], 4244, max_workers=1000)
