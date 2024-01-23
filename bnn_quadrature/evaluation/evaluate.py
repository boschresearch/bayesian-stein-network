import os
import time
from typing import Union

import torch

from bnn_quadrature.data.dataset_base import GenericDataClass
from bnn_quadrature.evaluation.plotting.plot_loss_runtime import PlotLoss
from bnn_quadrature.evaluation.plotting.plot_theta_time import ThetaPlot
from bnn_quadrature.models.gp import GPModel
from bnn_quadrature.models.nn.stein_model import SteinModel
from bnn_quadrature.options.device import my_device
from bnn_quadrature.options.enums import DeviceEnum
from bnn_quadrature.options.options import Options


def to_float(a):
    if isinstance(a, torch.Tensor):
        a = a.item()
    return a


def evaluate_training(
    model: Union[SteinModel, GPModel],
    integral: GenericDataClass,
    x: torch.Tensor,
    y: torch.Tensor,
    opts: Options,
    true_solution: Union[float, None],
):

    model.eval()
    plot_model(integral, model, opts, true_solution)
    theta_0, theta_dict, theta_mc = evaluate_theta_0(integral, model, opts)
    evaluate_std_theta_0(integral, model, opts, theta_0, theta_dict, theta_mc, x, y)


def evaluate_std_theta_0(integral, model, opts, theta_0, theta_dict, theta_mc, x, y):
    if opts.do_eval_uncertainty:
        t0 = time.time()
        my_device.device = DeviceEnum.cpu
        model.to(my_device.device)
        var_theta_model_scaled = model.get_theta_0_var(x=x.to(my_device.device), y=y.to(my_device.device), opts=opts)
        t1 = time.time()
        dt = t1 - t0
        torch.save(dt, os.path.join(opts.results_folder, "run_time_var.pt"))
        theta_dict["var_theta_model_scaled"] = to_float(var_theta_model_scaled)
        var_theta_model = integral.rescale_theta_var(var_theta_model_scaled)
        theta_dict["var_theta_model"] = to_float(var_theta_model)
        torch.save(theta_dict, os.path.join(opts.results_folder, "theta_final.pt"))
        print("--------------------------------------------")
        print(f"True solution:{integral.true_solution()}")
        print(f"BSN: {theta_0}")
        print(f"Std BSN: {torch.sqrt(var_theta_model)}")
        print(f"MC(MC): {theta_mc}")
        print("--------------------------------------------")


def evaluate_theta_0(integral, model, opts):
    state_dict = torch.load(
        os.path.join(str(opts.results_folder), f"model.pt"),
    )
    model.load_state_dict(state_dict)
    theta_0_scaled = model.get_theta_0().item()
    theta_mc_scaled = integral.get_theta_mc()
    theta_mc = integral.rescale_theta(theta_mc_scaled)
    theta_0 = integral.rescale_theta(theta_0_scaled)
    if not isinstance(theta_0, float):
        theta_0 = theta_0.flatten().item()
    print("--------------------------------------------")
    print(f"True solution:{integral.true_solution()}")
    print(f"BSN: {theta_0}")
    print(f"MC(MC): {theta_mc}")
    print("--------------------------------------------")
    theta_dict = {
        "theta_model": to_float(theta_0),
        "theta_mcmc": to_float(theta_mc),
        "theta_model_scaled": to_float(theta_0_scaled),
        "theta_mcmc_scaled": to_float(theta_mc_scaled),
    }
    torch.save(theta_dict, os.path.join(opts.results_folder, "theta_final.pt"))
    return theta_0, theta_dict, theta_mc


def plot_model(integral, model, opts, true_solution):
    if isinstance(model, SteinModel):
        plot = ThetaPlot(opts.name, data_dir=opts.results_folder)
        plot.generate_data(opts, true_solution, integral)
        plot.figure()

        plot = PlotLoss(opts.name, data_dir=opts.results_folder)
        plot.generate_data(opts)
        plot.figure()

