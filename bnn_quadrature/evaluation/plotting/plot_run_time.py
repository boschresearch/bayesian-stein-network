from typing import List

import torch

from bnn_quadrature.evaluation.plotting.plotting_main import settings, MethodEnum
from bnn_quadrature.evaluation.util.plotting_base import PlottingClass
from bnn_quadrature.evaluation.util.error_bar import plot_my_error
from bnn_quadrature.evaluation.util.folder import Folder, find_folder
from bnn_quadrature.evaluation.util.get_nn_result import get_nn_runtime, get_nn_runtime_var


class PlotRunTime(PlottingClass):
    NAME = "run_time"

    def generate_data(self, dim, folder_list: List[Folder], name):

        data_dict = {}
        for folder in folder_list:
            if folder.folder_name is not None:
                time, std_time, niter = get_nn_runtime(folder.folder_name)
                data_dict[folder.name] = {}
                data_dict[folder.name]["time"] = time
                data_dict[folder.name]["std_time"] = std_time
                data_dict[folder.name]["niter"] = niter
        torch.save(data_dict, self.data_file)

    def load(self):
        data_dict = torch.load(self.data_file)
        return data_dict

    def plot(self, ax, fig, legend=True, *args, **kwargs):
        data_dict = self.load()
        for method, method_dict in data_dict.items():
            time = method_dict["time"]
            std_time = method_dict["std_time"]
            niter = method_dict["niter"]
            plot_my_error(ax, niter, std_time, time)
        for method, method_dict in data_dict.items():
            niter = method_dict["niter"]
            time = method_dict["time"]
            ax.loglog(
                niter,
                time,
                **find_folder(method).get_settings(),
                **settings,
            )
        if legend:
            ax.legend()
        ax.grid()
        ax.set_xlabel(f"#Data Points")
        ax.set_ylabel("Runtime (s)")


class PlotRunTimeVar(PlottingClass):
    NAME = "run_time_var"

    def generate_data(self, dim, folder_list: List[Folder], name):

        data_dict = {}
        for folder in folder_list:
            if folder.folder_name is not None:
                time_var, std_time, niter = get_nn_runtime_var(folder.folder_name)
                data_dict[folder.name] = {}
                data_dict[folder.name]["time"] = time_var
                data_dict[folder.name]["std_time"] = std_time
                data_dict[folder.name]["niter"] = niter
        torch.save(data_dict, self.data_file)

    def load(self):
        data_dict = torch.load(self.data_file)
        return data_dict

    def plot(self, ax, fig, legend=True, *args, **kwargs):
        data_dict = self.load()
        for method, method_dict in data_dict.items():
            time = method_dict["time"]
            std_time = method_dict["std_time"]
            niter = method_dict["niter"]
            plot_my_error(ax, niter, std_time, time)
        for method, method_dict in data_dict.items():
            niter = method_dict["niter"]
            time = method_dict["time"]
            ax.loglog(
                niter,
                time,
                **find_folder(method).get_settings(),
                **settings,
            )
        if legend:
            ax.legend()
        ax.grid()
        ax.set_xlabel(f"#Data Points")
        ax.set_ylabel("Runtime (s)")


class PlotRunTimeMC(PlottingClass):
    NAME = "run_time_mc"

    def generate_data(
        self, dim, folder, folder_emu, folder_gp, folder_sgd, mc_time, name
    ):
        nn_time, nn_niter = get_nn_runtime(folder)
        sgd_time, sgd_niter = None, None
        gp_time, gp_niter = None, None
        if folder_sgd is not None:
            sgd_time, sgd_niter = get_nn_runtime(folder_sgd)
        if folder_gp is not None:
            gp_time, gp_niter = get_nn_runtime(folder_gp)
        bq_time, bq_niter = get_nn_runtime(folder_emu)
        data_dict = {
            "nn_time": nn_time,
            "nn_niter": nn_niter,
            "sgd_time": sgd_time,
            "sgd_niter": sgd_niter,
            "gp_time": gp_time,
            "gp_niter": gp_niter,
            "bq_time": bq_time,
            "bq_niter": bq_niter,
            "mc_time": mc_time,
        }
        torch.save(data_dict, self.data_file)

    def load(self):
        data_dict = torch.load(self.data_file)
        return (
            data_dict["nn_time"],
            data_dict["nn_niter"],
            data_dict["sgd_time"],
            data_dict["sgd_niter"],
            data_dict["gp_time"],
            data_dict["gp_niter"],
            data_dict["bq_time"],
            data_dict["bq_niter"],
            data_dict["mc_time"],
        )

    def plot(self, ax, fig, *args, **kwargs):
        (
            nn_time,
            nn_niter,
            sgd_time,
            sgd_niter,
            gp_time,
            gp_niter,
            bq_time,
            bq_niter,
            mc_time,
        ) = self.load()
        ax.loglog(
            nn_niter,
            nn_time + mc_time * nn_niter,
            **get_settings(MethodEnum.nn),
            **settings,
        )
        if sgd_niter is not None:
            ax.loglog(
                sgd_niter,
                sgd_time + mc_time * sgd_niter,
                **find_folder(MethodEnum.sgd).get_settings(),
                **settings,
            )
        ax.loglog(
            bq_niter,
            bq_time + bq_niter * mc_time,
            **get_settings(MethodEnum.bq),
            **settings,
        )
        ax.loglog(
            nn_niter, nn_niter * mc_time, **get_settings(MethodEnum.mc), **settings
        )
        if gp_niter is not None:
            ax.loglog(
                gp_niter,
                gp_time + mc_time * gp_niter,
                **get_settings(MethodEnum.gp),
                **settings,
            )
        ax.legend()
        ax.grid()
        ax.set_xlabel(f"#datapoints")
        ax.set_ylabel("run time")
