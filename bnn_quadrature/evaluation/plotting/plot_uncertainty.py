import warnings
from typing import List, Dict

import numpy as np
import torch
from matplotlib import pyplot as plt
from tueplots import cycler
from tueplots.constants.color import palettes

from bnn_quadrature.evaluation.plotting.plotting_main import MethodEnum, settings
from bnn_quadrature.evaluation.util.folder import Folder, find_folder
from bnn_quadrature.evaluation.util.plotting_base import PlottingClass


class PlotUncertainty(PlottingClass):
    NAME = "uncertainty"

    def generate_data(
        self,
        dim,
        folder_list: List[Folder],
        true_value,
        name,
    ):
        data_dict = {}
        for folder in folder_list:
            if folder.folder_name is not None:
                if folder.name == MethodEnum.nn or folder.name == MethodEnum.max:
                    niter, mean, std = get_std(folder.folder_name)
                    data_dict[folder.name] = {}
                    data_dict[folder.name]["std"] = std
                    data_dict[folder.name]["niter"] = niter
                    data_dict[folder.name]["mean"] = mean
                    if folder.name == MethodEnum.nn:
                        niter, mean = get_mean(folder.folder_name, name="theta_mcmc")
                        data_dict[MethodEnum.mc] = {}
                        data_dict[MethodEnum.mc]["niter"] = niter
                        data_dict[MethodEnum.mc]["mean"] = mean
                    if folder.name == MethodEnum.max:
                        niter, mean = get_mean(folder.folder_name, name="theta_mcmc")
                        data_dict[MethodEnum.mala] = {}
                        data_dict[MethodEnum.mala]["niter"] = niter
                        data_dict[MethodEnum.mala]["mean"] = mean
                else:
                    niter, mean = get_mean(folder.folder_name)
                    data_dict[folder.name] = {}
                    data_dict[folder.name]["niter"] = niter
                    data_dict[folder.name]["mean"] = mean
        data_dict["true_value"] = true_value
        torch.save(data_dict, self.data_file)

    def load(self):
        data_dict = torch.load(self.data_file)
        return data_dict

    def plot(self, ax, fig):
        data_dict = self.load()
        cycler_cycler = cycler.cycler(color=palettes.muted)
        plt.rcParams.update(cycler_cycler)
        true_value = data_dict["true_value"]
        data_dict.pop("true_value", None)
        try:
            nn_ = data_dict[MethodEnum.nn]
        except KeyError:
            nn_ = data_dict[MethodEnum.max]
        uncertainty_plot(ax, nn_, settings, true_value, MethodEnum.nn)
        for method, method_dict in data_dict.items():
            ax.semilogx(
                method_dict["niter"],
                method_dict["mean"],
                **find_folder(method).get_settings(),
                **settings,
            )
        ax.legend()
        ax.grid()
        ax.set_xlabel(f"#Data Points")
        ax.set_ylabel("$\\theta_0$")


def uncertainty_plot(ax, nn_dict: Dict, settings, true_value, folder_nn):
    niter_ = nn_dict["niter"]
    ax.hlines(
        true_value,
        niter_[0],
        niter_[-1],
        color="k",
        lw=0.7,
        linestyles="dashed",
    )
    mean_ = nn_dict["mean"]
    std_ = nn_dict["std"]
    color = find_folder(folder_nn).get_settings()["color"]
    ax.fill_between(
        niter_,
        mean_ - 3 * std_,
        mean_ + 3 * std_,
        alpha=0.2,
        facecolor=color,
    )
    ax.fill_between(
        niter_,
        mean_ - 2 * std_,
        mean_ + 2 * std_,
        alpha=0.2,
        facecolor=color,
    )
    ax.fill_between(
        niter_,
        mean_ - 1 * std_,
        mean_ + 1 * std_,
        alpha=0.2,
        facecolor=color,
    )


def get_std(folder):
    nn_array, niter_list = get_nn_result(folder)
    uncertainty, niter_list = get_uncertainty(folder)
    if (uncertainty < 0.0).any():
        warnings.warn(f"Negative uncertainty for for folder {folder}.")
        uncertainty = np.abs(uncertainty)

    std = np.sqrt(uncertainty)
    std = np.mean(std, axis=0)
    nn_mean = np.mean(nn_array, axis=0)
    nn_mean = nn_mean[: len(std)]
    return niter_list, nn_mean, std


def get_mean(folder, name="theta_model"):
    nn_array, niter_list = get_nn_result(folder, name=name)
    nn_mean = np.mean(nn_array, axis=0)
    return niter_list, nn_mean
