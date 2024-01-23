import os
import time

import torch
from torch.nn import Module
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from bnn_quadrature.extend_laplace_torch.jacobian import count_parameters
from bnn_quadrature.extend_laplace_torch.torch_hessian_laplace import (
    StableFullLaplace,
)
from bnn_quadrature.extend_laplace_torch.stein_ggn import SteinGGN


def fit_laplace(
    model: Module,
    x: torch.Tensor,
    y: torch.Tensor,
    dataset_size: int,
    results_folder: str,
):
    num_params = count_parameters(model)
    t0 = time.time()
    batch_size = dataset_size
    if dataset_size > num_params:
        batch_size = num_params
    train_loader = DataLoader(TensorDataset(x, y), batch_size=batch_size)
    n_epochs = 100
    la = StableFullLaplace(model, "regression", backend=SteinGGN)
    la.fit(train_loader)
    log_prior, log_sigma = (
        torch.ones(1, requires_grad=True),
        torch.ones(1, requires_grad=True),
    )
    hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-1)
    print("Optimizing hyper-parameters for Laplace approximation")
    for _ in tqdm(range(n_epochs)):
        hyper_optimizer.zero_grad()
        neg_marglik = -la.log_marginal_likelihood(log_prior.exp(), log_sigma.exp())
        neg_marglik.backward()
        hyper_optimizer.step()
    print("=============================")
    print(time.time()-t0)
    print("=============================")
    torch.save(la, os.path.join(results_folder, "la.pt"))
