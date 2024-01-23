import os

import torch

from bnn_quadrature.evaluation.util.plotting_base import PlottingClass


class PlotLoss(PlottingClass):
    NAME = "loss"

    def generate_data(self, opts):
        loss = torch.load(os.path.join(opts.results_folder, "loss_list.pt"))
        data_dict = {
            "loss": loss,
        }
        torch.save(data_dict, self.data_file)

    def load(self):
        return torch.load(self.data_file)["loss"]

    def plot(self, ax, fig, *args, **kwargs):
        loss_list = self.load()
        ax.plot(loss_list, lw=0.7)
        ax.set_xlabel(f"Iteration")
        ax.set_ylabel(f"log(loss)")
        ax.grid()
