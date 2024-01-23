import os

import torch

from bnn_quadrature.evaluation.util.plotting_base import PlottingClass


class ThetaPlot(PlottingClass):
    NAME = "theta"

    def generate_data(self, opts, true_solution, integral):
        theta_dict = torch.load(os.path.join(opts.results_folder, "theta_dict.pt"))
        theta = integral.rescale_theta(torch.tensor(theta_dict["mean"]))
        iter = theta_dict["iter"]
        data_dict = {
            "theta_dict": theta,
            "iter": iter,
            "true_solution": true_solution,
        }
        torch.save(data_dict, self.data_file)

    def load(self):
        data_dict = torch.load(self.data_file)
        return data_dict["theta_dict"], data_dict["iter"], data_dict["true_solution"]

    def plot(self, ax, fig, *args, **kwargs):
        m, iter, true_theta = self.load()
        ax.plot(iter, m, lw=0.7)
        if true_theta is not None:
            ax.axhline(y=true_theta, color="k", ls="-.", lw=0.7)
        ax.grid()
        ax.set_xlabel(f"Iteration")
        ax.set_ylabel(f"$\\theta_0$")
