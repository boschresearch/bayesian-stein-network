import math
import os
import time
from typing import Dict, List

import numpy as np
import torch
from hessianfree.optimizer import HessianFree
from torch.nn import MSELoss
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from bnn_quadrature.models.nn.stein_model import SteinModel
from bnn_quadrature.options.enums import PytorchSolverEnum
from bnn_quadrature.options.options import Options


def find_optimizer(solver: PytorchSolverEnum, model, opts):
    if solver == PytorchSolverEnum.hessian_free:
        optim = HessianFree
        optimizer = optim(model.parameters())
        return optimizer
    elif solver == PytorchSolverEnum.pt_l_bfgs:
        optim = torch.optim.LBFGS
        optimizer = optim(
            model.parameters(),
            lr=1.0,
            line_search_fn="strong_wolfe",
            max_iter=opts.max_iter,
            history_size=100,
            tolerance_grad=1e-15,
            tolerance_change=1e-15,
        )
        return optimizer
    elif solver == PytorchSolverEnum.adam:
        optim = torch.optim.Adam
    elif solver == PytorchSolverEnum.sgd:
        optim = torch.optim.SGD
    else:
        raise ValueError(f"Solver does not exist {solver}")

    optimizer = optim(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
    return optimizer


def training(model: SteinModel, data_loader: DataLoader, opts: Options):
    torch.cuda.empty_cache()
    theta_dict = {"mean": [], "var": [], "iter": []}  # type: Dict[str, List[float]]
    loss_list = []
    loss_function = MSELoss()
    optimizer = find_optimizer(opts.solver, model, opts)
    print("Starting training ...")
    t0 = time.time()
    inputs = (
        model,
        data_loader,
        optimizer,
        loss_function,
        opts,
        loss_list,
        theta_dict,
    )
    if opts.solver == PytorchSolverEnum.pt_l_bfgs:
        training_bfgs(*inputs)
    elif opts.solver == PytorchSolverEnum.hessian_free:
        training_hessian_free(*inputs)
    else:
        training_standard(*inputs)
    t1 = time.time()
    dt = t1 - t0
    torch.save(dt, os.path.join(opts.results_folder, "run_time.pt"))
    torch.save(theta_dict, os.path.join(opts.results_folder, "theta_dict.pt"))
    torch.save(loss_list, os.path.join(opts.results_folder, "loss_list.pt"))
    print("finished training")
    torch.save(model.state_dict(), os.path.join(str(opts.results_folder), f"model.pt"))
    print("Finished training")


def training_bfgs(
    model, data_loader, optimizer, loss_function, opts, loss_list, theta_dict
):
    for _, data in enumerate(data_loader):
        x, y = data
        global iteration
        iteration = 0

        def closure():
            global iteration
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_function(logits, y)
            if opts.weight_decay > 0.0:
                loss = apply_weight_decay(model, loss, opts.weight_decay)
            loss.backward()
            loss_list.append(np.log10(loss.cpu().detach().numpy()))
            theta_dict["mean"].append(model.get_theta_0().item())
            theta_dict["iter"].append(iteration)
            iteration += 1
            return loss

        optimizer.step(closure)


def training_hessian_free(
    model, data_loader, optimizer, loss_function, opts, loss_list, theta_dict
):
    epochs = math.ceil(opts.max_iter / int(opts.dataset_size / opts.batch_size))
    for iteration in range(epochs):
        for _, data in enumerate(data_loader):
            x, y = data

            def closure():
                optimizer.zero_grad()
                logits = model(x)
                loss = loss_function(logits, y)
                if opts.weight_decay > 0.0:
                    loss = apply_weight_decay(model, loss, opts.weight_decay)
                return loss, logits

            optimizer.step(forward=closure)
            with torch.no_grad():
                logits = model(x)
                loss = loss_function(logits, y)
            loss_list.append(np.log10(loss))
            theta_dict["mean"].append(model.get_theta_0().item())
            theta_dict["iter"].append(iteration)


def training_standard(
    model, data_loader, optimizer, loss_function, opts, loss_list, theta_dict
):
    epochs = math.ceil(opts.max_iter / int(opts.dataset_size / opts.batch_size))
    for iteration in range(epochs):
        for _, data in enumerate(data_loader):
            x, y = data
            loss = iterate_one_training_step(
                model=model,
                x=x,
                y=y,
                optimizer=optimizer,
                loss_function=loss_function,
                opts=opts,
            )
        if iteration % 10 == 0:
            print(f"Iteration: {iteration}, loss: {loss}")
        loss_list.append(np.log10(loss))
        theta_dict["mean"].append(model.get_theta_0().item())
        theta_dict["iter"].append(iteration)


def iterate_one_training_step(
    model: SteinModel,
    x: torch.Tensor,
    y: torch.Tensor,
    optimizer: Optimizer,
    loss_function: _Loss,
    opts: Options,
):

    optimizer.zero_grad()
    logits = model(x)
    loss = loss_function(logits, y)
    if opts.weight_decay > 0.0:
        loss = apply_weight_decay(model, loss, opts.weight_decay)
    loss.backward()
    optimizer.step()
    return loss.detach().numpy()


def apply_weight_decay(model, loss, weight_decay):
    params = list(model.parameters())
    params_list = []
    for p in params:
        params_list.append(p.flatten())
    params = torch.cat(params_list)
    param_norm = torch.norm(params)
    loss = loss + weight_decay * param_norm
    return loss
