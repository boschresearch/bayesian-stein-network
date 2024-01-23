from typing import List

import torch

from bnn_quadrature.evaluation.plotting.plotting_main import settings, MethodEnum
from bnn_quadrature.evaluation.util.plotting_base import PlottingClass
from bnn_quadrature.evaluation.util.folder import Folder, find_folder
from bnn_quadrature.evaluation.util.get_nn_result import get_nn_runtime, get_nn_result
from bnn_quadrature.evaluation.util.get_error import get_error


class PlotRuntimeError(PlottingClass):
    NAME = "runtime_rmse"

    def generate_data(self, dim, folder_list: List[Folder], true_value, name):
        data_dict = {}
        for folder in folder_list:
            if folder.folder_name is not None:
                time, _, _ = get_nn_runtime(folder.folder_name)
                array, niter_list = get_nn_result(folder.folder_name)
                _, error, _ = get_error(array, true_value)
                data_dict[folder.name] = {}
                data_dict[folder.name]["time"] = time
                data_dict[folder.name]["error"] = error

        torch.save(data_dict, self.data_file)

    def load(self):
        data_dict = torch.load(self.data_file)
        return data_dict

    def plot(self, ax, fig, legend=True, *args, **kwargs):
        data_dict = self.load()
        for method, method_dict in data_dict.items():
            ax.loglog(
                method_dict["time"],
                method_dict["error"],
                **find_folder(method).get_settings(),
                **settings,
            )
        if legend:
            ax.legend()
        ax.grid()
        ax.set_xlabel(f"Runtime (s)")
        ax.set_ylabel("Mean Rel. Error")


class PlotRuntimeRMSEMC(PlottingClass):
    NAME = "runtime_rmse_mc"

    def generate_data(self, dim, folder_list, true_value, mc_time, name):
        data_dict = {}
        for folder in folder_list:
            if folder.folder_name is not None:
                time, _, _ = get_nn_runtime(folder.folder_name)
                print(time)
                nn_array, niter_list = get_nn_result(folder.folder_name)
                _, error, _ = get_error(nn_array, true_value)
                time += niter_list * mc_time
                data_dict[folder.name] = {}
                data_dict[folder.name]["time"] = time
                data_dict[folder.name]["error"] = error
                if folder.name == MethodEnum.nn or folder.name == MethodEnum.max:
                    nn_array, niter_list = get_nn_result(
                        folder.folder_name, name="theta_mcmc"
                    )
                    _, error, _ = get_error(nn_array, true_value)
                    time = niter_list * mc_time
                    data_dict[MethodEnum.mc] = {}
                    data_dict[MethodEnum.mc]["time"] = time
                    data_dict[MethodEnum.mc]["error"] = error
        torch.save(data_dict, self.data_file)

    def load(self):
        data_dict = torch.load(self.data_file)
        return data_dict

    def plot(self, ax, fig, *args, **kwargs):
        data_dict = self.load()
        for method, method_dict in data_dict.items():
            ax.loglog(
                method_dict["time"],
                method_dict["error"],
                **find_folder(method).get_settings(),
                **settings,
            )
        ax.legend()
        ax.grid()
        ax.set_xlabel(f"Run Time (incl. sampling) [s]")
        ax.set_ylabel("Mean Abs. Error")
