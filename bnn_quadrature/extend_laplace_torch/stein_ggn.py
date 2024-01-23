import time

import torch
from laplace.curvature import GGNInterface

from bnn_quadrature.extend_laplace_torch.jacobian import compute_jacobian

"""
The following function is adapted from Laplace Version 0.1a1
( https://github.com/AlexImmer/Laplace/releases/tag/0.1a1
Copyright (c) 2021 Alex Immer, licensed under the MIT License,
cf. 3rd-party-license.txt file in the root directory of this source tree)
to produce GGN approximation for the Stein network.
"""


class SteinGGN(GGNInterface):
    """
    Extension of laplace-torch to use own implementation of Jacobian.
    This is necessary since backpack does not support neural ODEs.
    """

    def kron(self, x, y, **kwargs):
        raise NotImplementedError(
            "Kronecker-factorized Hessian not yet implemented for SteinGGN"
        )

    def diag(self, x, y, **kwargs):
        raise NotImplementedError("Diagonal Hessian not yet implemented for SteinGGN")

    def jacobians(self, x: torch.Tensor):
        t0 = time.time()
        jacobian = compute_jacobian(self.model, x)
        print(time.time() - t0)
        return jacobian

    def gradients(self, x, y):
        f = self.model(x)
        loss = self.lossfunc(f, y)
        loss.backward()
        Gs = torch.cat(
            [p.grad_batch.data.flatten(start_dim=1) for p in self._model.parameters()],
            dim=1,
        )
        return Gs, loss
