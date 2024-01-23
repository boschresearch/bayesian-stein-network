from typing import List

import torch

from bnn_quadrature.evaluation.plotting.plotting_main import (
    settings,
)
from bnn_quadrature.evaluation.util.plotting_base import PlottingClass
from bnn_quadrature.evaluation.util.folder import Folder, find_folder
from bnn_quadrature.evaluation.util.get_nn_result import get_loss


class PlotLossComparison(PlottingClass):
    NAME = "loss"

    def generate_data(self, folder_list: List[Folder], name):
        data_dict = {}
        for folder in folder_list:
            if folder.folder_name is not None:
                loss, niter = get_loss(folder.folder_name)
                data_dict[folder.name] = {}
                data_dict[folder.name]["niter"] = niter
                data_dict[folder.name]["loss"] = loss
                print(loss)
        torch.save(data_dict, self.data_file)

    def load(self):
        data_dict = torch.load(self.data_file)
        return data_dict

    def plot(self, ax, fig, *args, **kwargs):
        data_dict = self.load()
        for method, method_dict in data_dict.items():
            ax.semilogx(
                method_dict["niter"],
                method_dict["loss"],
                **find_folder(method).get_settings(),
                **settings,
            )
        ax.legend()
        ax.grid()
        ax.set_xlabel(f"#Data Points")
        ax.set_ylabel("Loss")
