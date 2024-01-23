import torch


def calculate_dk_dx1_rbf(k, x1, x2, l):
    """
    :param k: Kernel of shape n1 x n2
    :param x1: Tensor of shape n1 x 1 x d
    :param x2: Tensor of shape n2 x 1 x d
    :param l:
    :return:  Tensor of shape n1 x n2 x d
    """
    return (-(x1 - x2.transpose(0, 1)) / l ** 2) * k.unsqueeze(-1)


def calculate_dk_dx2_rbf(k, x1, x2, l):
    """
    :param k: Kernel of shape n1 x n2
    :param x1: Tensor of shape n1 x 1 x d
    :param x2: Tensor of shape n2 x 1 x d
    :param l:
    :return:  Tensor of shape n1 x n2 x d
    """
    return ((x1 - x2.transpose(0, 1)) / l ** 2) * k.unsqueeze(-1)


def calculate_dk_dx1_dx2_rbf(k, x1, x2, l):
    """
    :param k: Kernel of shape n1 x n2
    :param x1: Tensor of shape n1 x 1 x d
    :param x2: Tensor of shape n2 x 1 x d
    :param l:
    :return:  Tensor of shape n1 x n2
    """
    dx = -(x1 - x2.transpose(0, 1))
    dx = torch.einsum('bcd, bcd ->bc', dx, dx)
    return -(dx / l ** 4) * k + k / l ** 2
