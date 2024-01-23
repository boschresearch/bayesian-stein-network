import torch

from bnn_quadrature.data.pdf.pdf import (
    NormalPDF,
)


def test_grad_log_pi():
    pdf = NormalPDF()
    x = torch.tensor([[1.0], [2.0], [0]]).unsqueeze(1)
    assert torch.Size([x.shape[0], 1, 1]) == pdf.grad_log(x).shape
