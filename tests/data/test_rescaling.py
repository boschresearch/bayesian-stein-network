import torch

from bnn_quadrature.data.dataset_base import DataClass


def test_rescaling():
    dim = 13
    dataset_size = 40
    x = torch.randn([dataset_size, 1, dim])
    integral = DataClass(dim=dim, use_y_rescaling=False, use_x_rescaling=True, dataset_size=10, version=0)
    integral.use_x_rescaling = True
    integral.mean_x, integral.std_x = integral.get_rescaling_factor(x)
    x_rescaled = integral.rescale_x(x)
    x_undo_rescale = integral.pdf.undo_rescale(x_rescaled)
    assert (x == x_undo_rescale).all
