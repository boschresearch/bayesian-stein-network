import torch
from laplace import FullLaplace

from bnn_quadrature.extend_laplace_torch.stable_cholesky import stable_cholesky


"""
The following function is adapted from Laplace Version 0.1a1
( https://github.com/AlexImmer/Laplace/releases/tag/0.1a1
Copyright (c) 2021 Alex Immer, licensed under the MIT License,
cf. 3rd-party-license.txt file in the root directory of this source tree)
to extend the Laplace approximation.
"""


class StableFullLaplace(FullLaplace):
    """
    Overwrite _compute_scale to add a jitter value to the computation of ``M^{-0.5}``.
    """

    def _compute_scale(self):
        self._posterior_scale = invsqrt_precision(self.posterior_precision)


def _precision_to_scale_tril(P):
    # Ref: https://nbviewer.jupyter.org/gist/fehiepsi/5ef8e09e61604f10607380467eb82006#Precision-to-scale_tril
    flipped_P = torch.flip(P, (-2, -1))

    Lf = stable_cholesky(flipped_P)

    L_inv = torch.transpose(torch.flip(Lf, (-2, -1)), -2, -1)
    L = torch.linalg.solve_triangular(
        L_inv, torch.eye(P.shape[-1], dtype=P.dtype, device=P.device), upper=False
    )
    return L


def invsqrt_precision(M):
    """Compute ``M^{-0.5}`` as a tridiagonal matrix.

    Parameters
    ----------
    M : torch.Tensor

    Returns
    -------
    M_invsqrt : torch.Tensor
    """
    return _precision_to_scale_tril(M)

