from typing import Callable

import gpytorch
import torch

from bnn_quadrature.models.gp.rbf_stein_kernel import (
    calculate_dk_dx1_rbf,
    calculate_dk_dx2_rbf,
    calculate_dk_dx1_dx2_rbf,
)


class SteinKernelRBF(gpytorch.kernels.Kernel):
    def __init__(self, kernel: gpytorch.kernels.ScaleKernel, grad_log_p: Callable):
        super(SteinKernelRBF, self).__init__()
        """
        Warning: Be careful which Kernels one uses - some of the Lazy tensors do some weird stuff in memory...
        """
        self.kernel = kernel
        self.grad_log_p = grad_log_p

    def get_lengthscale(self):
        return self.kernel.base_kernel.lengthscale

    def forward(self, x1, x2, **params):
        k = self.kernel(x1, x2).evaluate()
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
        lengthscale = self.get_lengthscale()
        dk_dx1 = calculate_dk_dx1_rbf(k, x1, x2, lengthscale)
        dk_dx2 = calculate_dk_dx2_rbf(k, x1, x2, lengthscale)
        dk_dx1_dx2 = calculate_dk_dx1_dx2_rbf(
            k, x1, x2, lengthscale
        )
        dlog_p_dx2 = self.grad_log_p(x2)
        dlog_p_dx1 = self.grad_log_p(x1)
        dk_dx1_dlog_p_dx2 = torch.einsum("bcd, cd -> bc", dk_dx1, dlog_p_dx2.squeeze(1))
        dk_dx2_dlog_p_dx1 = torch.einsum("bcd, bd -> bc", dk_dx2, dlog_p_dx1.squeeze(1))
        dlog_p_dx1_dlog_p_dx2 = torch.einsum("bcd, gcd -> bg", dlog_p_dx1, dlog_p_dx2)
        sk = (
            dk_dx1_dx2
            + dk_dx1_dlog_p_dx2
            + dk_dx2_dlog_p_dx1
            + k * dlog_p_dx1_dlog_p_dx2
        )

        return sk
