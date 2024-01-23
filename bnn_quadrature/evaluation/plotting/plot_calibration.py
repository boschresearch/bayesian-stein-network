from typing import List

import numpy as np
import torch

from bnn_quadrature.evaluation.plotting.plotting_main import (
    MethodEnum,
    settings,
)
from bnn_quadrature.evaluation.util.plotting_base import PlottingClass
from bnn_quadrature.evaluation.util.error_bar import plot_my_error
from bnn_quadrature.evaluation.util.folder import Folder, find_folder
from bnn_quadrature.evaluation.util.get_nn_result import get_uncertainty, get_nn_result


class PlotCalibration(PlottingClass):
    NAME = "calibration"

    def generate_data(
        self,
        dim,
        folder_list: List[Folder],
        true_value,
        name,
        plot_bq_uncertainty=False,
    ):

        data_dict = {}
        for folder in folder_list:
            if folder.folder_name is not None:
                niter, calibration, std_calibration = get_calibration(folder.folder_name, true_value)
                data_dict[folder.name] = {}
                data_dict[folder.name]["niter"] = niter
                data_dict[folder.name]["calibration"] = calibration
                data_dict[folder.name]["std_calibration"] = std_calibration
        torch.save(data_dict, self.data_file)

    def load(self):
        data_dict = torch.load(self.data_file)
        return data_dict

    def plot(self, ax, fig, lolims=True, uplims=False, *args, **kwargs):
        data_dict = self.load()
        for method, method_dict in data_dict.items():
            if method is not MethodEnum.gp:
                niter = method_dict["niter"]
                calibration = method_dict["calibration"]
                std_calibration = method_dict["std_calibration"]
                plot_my_error(ax, niter, std_calibration, calibration, lolims, uplims)
        for method, method_dict in data_dict.items():
            if method is not MethodEnum.gp:
                ax.semilogx(
                    method_dict["niter"],
                    method_dict["calibration"],
                    **find_folder(method).get_settings(),
                    **settings,
                )
        ax.legend()
        ax.grid()
        ax.set_xlabel(f"#Data Points")
        ax.set_ylabel("Calibration")


def get_calibration(folder, true_value):
    nn_array, niter_list = get_nn_result(folder, name="theta_model")
    uncertainty, niter_list = get_uncertainty(folder)
    nn_array = nn_array[:, 0: np.shape(uncertainty)[-1]]
    calibration = np.abs(nn_array - true_value) / np.sqrt(uncertainty)
    calibration_mean = np.mean(calibration, axis=0)
    calibration_std = np.std(calibration, axis=0)
    print(calibration_mean)
    return niter_list, calibration_mean, calibration_std
