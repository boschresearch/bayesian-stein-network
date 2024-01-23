import torch

from bnn_quadrature.data.pdf.pdf import PDF
from bnn_quadrature.models.nn.stein_layer import (
    calculate_div_u,
    calculate_grad_log_pi_u
)


def test_calculate_div_u():
    a = 1.0
    b1 = 2.0
    b2 = 3.0
    c11 = 4.0
    c12 = 5.0
    c22 = 6.0

    def u(x):
        x1 = x[:, :, 0]
        x2 = x[:, :, 1]
        u = torch.zeros_like(x)
        u[:, :, 0] = (
            a + b1 * x1 + b2 * x2 + c11 * x1 * x1 + c12 * x1 * x2 + c22 * x2 * x2
        )
        u[:, :, 1] = (
            a + b1 * x1 + b2 * x2 + c11 * x1 * x1 + c12 * x1 * x2 + c22 * x2 * x2
        )
        return u

    def true_div_u(x):
        x1 = x[:, :, 0]
        x2 = x[:, :, 1]
        du1_dx1 = b1 + 2 * x1 * c11 + c12 * x2
        du2_dx2 = b2 + 2 * x2 * c22 + c12 * x1
        return du1_dx1 + du2_dx2

    x = torch.tensor([[1.0, 1.0], [2.0, 4.0], [0, 0]]).unsqueeze(1)
    assert x.shape == u(x).shape
    div_u = calculate_div_u(u, x)
    assert div_u.shape == x.shape[0:1]
    true_div_u = true_div_u(x).flatten()
    assert torch.allclose(div_u, true_div_u)


def test_calculate_grad_log_pi_u():
    a = 1.0
    b1 = 2.0
    b2 = 3.0
    c11 = 4.0
    c12 = 5.0
    c22 = 6.0

    def u(x):
        return torch.ones_like(x)

    def log_pi(x):
        x1 = x[:, :, 0]
        x2 = x[:, :, 1]
        return a + b1 * x1 + b2 * x2 + c11 * x1 * x1 + c12 * x1 * x2 + c22 * x2 * x2

    def true_grad_log_pi_u(x):
        x1 = x[:, :, 0]
        x2 = x[:, :, 1]
        return b1 + b2 + 2 * c11 * x1 + c12 * x1 + c12 * x2 + 2 * c22 * x2

    pdf = PDF()
    pdf.log_pdf = log_pi
    x = torch.tensor([[1.0, 1.0], [2.0, 4.0], [0, 0]]).unsqueeze(1)
    grad_log_pi = pdf.grad_log(x)
    grad_log_pi_u = calculate_grad_log_pi_u(grad_log_pi, u, x)
    true_grad_log_pi_u = true_grad_log_pi_u(x).flatten()
    assert torch.allclose(grad_log_pi_u, true_grad_log_pi_u)
