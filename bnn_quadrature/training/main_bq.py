import os
import time
import warnings

import GPy
import numpy as np
import torch
from emukit.model_wrappers.gpy_quadrature_wrappers import BaseGaussianProcessGPy, RBFGPy
from emukit.quadrature.methods import VanillaBayesianQuadrature

from bnn_quadrature.data.pdf.pdf import StandardNormalPDF
from bnn_quadrature.extend_emukit.quadrature_rbf_truncated_gaussian import (
    QuadratureRBFTruncatedGaussian,
    normalization_trunc_gauss,
)
from bnn_quadrature.training.main_bsn import initialize_dataset
from bnn_quadrature.options.options import Options


def run_bq(opts: Options):
    if opts.use_x_rescaling:
        warnings.warn(
            "x-rescaling not implemented for Bayesian quadrature"
            "Setting use_x_rescaling to false."
        )
        opts.use_x_rescaling = False
    integral = initialize_dataset(opts)
    x, y = get_dataset(integral, opts)

    t0 = time.time()
    theta_0_scaled, theta_0_var_scaled, my_measure, emukit_method = calculate_theta(
        x, y, integral
    )
    t1 = time.time()
    dt = t1 - t0
    torch.save(dt, os.path.join(opts.results_folder, "run_time.pt"))
    theta_0_var = integral.rescale_theta_var(theta_0_var_scaled)

    theta_0 = integral.rescale_theta(theta_0_scaled)
    theta_mc_scaled = integral.get_theta_mc()
    theta_mc = integral.rescale_theta(theta_mc_scaled)
    print("--------------------------------------------")
    print(f"True solution:{integral.true_solution()}")
    print(f"BQ: {theta_0}")
    if isinstance(integral.pdf, StandardNormalPDF):
        print(f"Std BQ: {np.sqrt(theta_0_var)}")
    print(f"MC: {theta_mc}")
    print("--------------------------------------------")
    theta_dict = {
        "theta_model": theta_0,
        "theta_model_scaled": theta_0_scaled,
        "var_theta_model_scaled": theta_0_var_scaled,
        "var_theta_model": theta_0_var
    }
    torch.save(theta_dict, os.path.join(opts.results_folder, "theta_final.pt"))


def calculate_theta(x, y, integral):
    theta_0_scaled = 0.0
    theta_0_var_scaled = 0.0
    normalization = 0.0
    list_of_measures, weights = integral.emukit_pdf
    for my_measure, weight in zip(list_of_measures, weights):
        z = calculate_normalization(integral, my_measure)

        emukit_method, integral_mean, integral_var = run_quadrature(
            x, y, my_measure
        )

        theta_0_scaled += weight * integral_mean * z
        theta_0_var_scaled = integral_var
        normalization += weight * z
    theta_0_scaled = float(theta_0_scaled / normalization)
    return theta_0_scaled, theta_0_var_scaled, my_measure, emukit_method


def calculate_normalization(integral, my_measure):
    if my_measure.mean.shape[-1] > 1:
        z = 1.0
        for a, b, mean, var in zip(
            my_measure.a, my_measure.b, my_measure.mean, my_measure.variance
        ):
            z *= normalization_trunc_gauss(a, b, mean, np.sqrt(var))
    else:
        z = normalization_trunc_gauss(
            integral.a, integral.b, my_measure.mean, np.sqrt(my_measure.variance)
        )
    return z


def run_quadrature(X_init, Y_init, my_measure):
    gpy_model = GPy.models.GPRegression(
        X=X_init,
        Y=Y_init,
        kernel=GPy.kern.RBF(input_dim=X_init.shape[1], lengthscale=1.0, variance=1.0),
    )
    gpy_model.optimize()
    emukit_rbf = RBFGPy(gpy_model.kern)
    emukit_qrbf = QuadratureRBFTruncatedGaussian(emukit_rbf, measure=my_measure)
    emukit_model = BaseGaussianProcessGPy(kern=emukit_qrbf, gpy_model=gpy_model)
    emukit_method = VanillaBayesianQuadrature(base_gp=emukit_model, X=X_init, Y=Y_init)
    initial_integral_mean, initial_integral_variance = emukit_method.integrate()
    return emukit_method, initial_integral_mean, initial_integral_variance


def get_dataset(integral, opts: Options):
    x, y= integral.get_dataset(n_max=opts.dataset_size)
    x = x.squeeze(1).numpy()
    y = y.numpy()
    return x, y
