from typing import List

import torch

from bnn_quadrature.evaluation.plotting.plotting_main import (
    MethodEnum,
    settings,
)
from bnn_quadrature.evaluation.util.plotting_base import PlottingClass
from bnn_quadrature.evaluation.util.error_bar import plot_my_error
from bnn_quadrature.evaluation.util.folder import Folder, find_folder
from bnn_quadrature.evaluation.util.get_nn_result import get_nn_result
from bnn_quadrature.evaluation.util.get_error import get_error


class PlotComparison(PlottingClass):
    NAME = "comparison"

    def generate_data(self, dim, folder_list: List[Folder], true_value, name):
        data_dict = {}
        for folder in folder_list:
            if folder.folder_name is not None:
                array, niter = get_nn_result(folder.folder_name)
                _, error, std_error = get_error(array, true_value)
                data_dict[folder.name] = {}
                data_dict[folder.name]["niter"] = niter
                data_dict[folder.name]["error"] = error
                data_dict[folder.name]["std_error"] = std_error
            if folder.name == MethodEnum.nn:
                array, niter = get_nn_result(folder.folder_name, name="theta_mcmc")
                _, error, std_error = get_error(array, true_value)
                data_dict[MethodEnum.mc] = {}
                data_dict[MethodEnum.mc]["niter"] = niter
                data_dict[MethodEnum.mc]["error"] = error
                data_dict[MethodEnum.mc]["std_error"] = std_error
            if folder.name == MethodEnum.qmc:
                array, niter = get_nn_result(folder.folder_name, name="theta_mcmc")
                _, error, std_error = get_error(array, true_value)
                data_dict[MethodEnum.qmc2] = {}
                data_dict[MethodEnum.qmc2]["niter"] = niter
                data_dict[MethodEnum.qmc2]["error"] = error
                data_dict[MethodEnum.qmc2]["std_error"] = std_error
            if folder.name == MethodEnum.max:
                array, niter = get_nn_result(folder.folder_name, name="theta_mcmc")
                _, error, std_error = get_error(array, true_value)
                data_dict[MethodEnum.mala] = {}
                data_dict[MethodEnum.mala]["niter"] = niter
                data_dict[MethodEnum.mala]["error"] = error
                data_dict[MethodEnum.mala]["std_error"] = std_error

        torch.save(data_dict, self.data_file)

    def load(self):
        data_dict = torch.load(self.data_file)
        return data_dict

    def plot(self, ax, fig, legend=True, plot_error=True, *args, **kwargs):
        data_dict = self.load()
        ax.set_xscale("log")
        ax.set_yscale("log")
        if plot_error:
            for method, method_dict in data_dict.items():
                niter = method_dict["niter"]
                error = method_dict["error"]
                std_error = method_dict["std_error"]
                plot_my_error(ax, niter, std_error, error)
        for method, method_dict in data_dict.items():
            niter = method_dict["niter"]
            error = method_dict["error"]
            ax.loglog(
                niter,
                error,
                **find_folder(method).get_settings(),
                **settings,
            )
        if legend:
            ax.legend()
        ax.grid()
        ax.set_xlabel(f"#Data Points")
        ax.set_ylabel("Mean Rel. Error")
