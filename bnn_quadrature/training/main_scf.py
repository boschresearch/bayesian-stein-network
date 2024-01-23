import os
import time

import gpytorch
import torch
from gpytorch.likelihoods import Likelihood

from bnn_quadrature.data.pdf.pdf import PDF
from bnn_quadrature.evaluation.evaluate import evaluate_training
from bnn_quadrature.options import Options
from bnn_quadrature.training.main_bsn import initialize_dataset
from bnn_quadrature.models.gp import find_kernel, GPModel, SteinGPModel, SteinKernelRBF


def run_scf(opts: Options):
    integral = initialize_dataset(opts)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    x, y = integral.get_dataset(n_max=opts.dataset_size)
    x = x.squeeze(1)
    y = y.squeeze(1)
    model = initialize_gp_model(opts, integral.pdf, x, y, likelihood)
    t0 = time.time()
    train_model = True
    if train_model:
        for i in range(opts.max_iter):
            model = initialize_gp_model(opts, integral.pdf, x, y, likelihood)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
            optimizer.zero_grad()
            loss = -model.mll(x, y)
            print(
                f"Loss: {loss}, l: {model.gp.sk.get_lengthscale().item()}, noise: {likelihood.noise_covar.noise.item()}"
            )
            loss.backward()
            optimizer.step()
            noise = likelihood.noise_covar.noise
            opts.hps[opts.noise_key] = noise
            lengthscale = model.gp.sk.get_lengthscale()
            opts.hps[opts.lengthscale_key] = lengthscale
    t1 = time.time()
    dt = t1 - t0
    torch.save(dt, os.path.join(opts.results_folder, "run_time.pt"))
    torch.save(model.state_dict(), os.path.join(str(opts.results_folder), f"model.pt"))
    evaluate_training(
        model,
        x=x,
        y=y,
        opts=opts,
        integral=integral,
        true_solution=integral.true_solution(),
    )


def initialize_gp_model(
    opts: Options, pdf: PDF, x: torch.Tensor, y: torch.Tensor, likelihood: Likelihood
) -> GPModel:
    kernel = find_kernel(opts.kernel)
    stein_kernel = SteinKernelRBF(kernel, grad_log_p=pdf.grad_log)
    gp = SteinGPModel(x, y, likelihood, kernel=stein_kernel)
    model = GPModel(gp=gp, train_x=x, train_y=y, likelihood=likelihood)
    model.initialize(**opts.hps)
    return model
