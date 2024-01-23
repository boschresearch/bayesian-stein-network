import os
import warnings
from abc import ABC
from pathlib import Path
from typing import Union, Tuple

import numpy as np
import torch

from bnn_quadrature.data.pdf.pdf import (
    StandardNormalPDF,
)
from bnn_quadrature.data.pdf_bq.gaussian import standard_normal


class GenericDataClass(ABC):
    visualization_dim: int
    NAME = ""
    dim = 1

    def __init__(
        self,
        use_x_rescaling: bool = False,
        use_y_rescaling: bool = False,
        *args,
        **kwargs,
    ):
        super(GenericDataClass, self).__init__()
        self.a = None
        self.b = None
        self.index_of_bounds = None
        warnings.warn(f"Not using {args}, {kwargs}")
        self.dataset_path = str(Path(__file__).parent.parent / "datasets")
        self.name = f"{self.NAME}"
        self.use_x_rescaling = use_x_rescaling
        self.use_y_rescaling = use_y_rescaling
        self.mean_x = None
        self.std_x = None
        self.mean_y = None
        self.std_y = None
        self.pdf = None
        self.emukit_pdf = None
        self.n_max = None

    def get_rescaling_factor(self, x) -> Union[Tuple[torch.Tensor, torch.Tensor], None]:
        if self.use_x_rescaling:
            mean = torch.mean(x, dim=0).squeeze(0)
            std = torch.std(x, dim=0).squeeze(0)
            return mean, std

    def f(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def true_solution(self):
        pass

    def rescale_y(self, y):
        y_rescaling_factor = torch.max(y) - torch.min(y)
        y_min = torch.min(y)
        y = (y - y_min) / y_rescaling_factor
        self.mean_y = y_min
        self.std_y = y_rescaling_factor
        return y

    def rescale_x(self, x):
        self.pdf.update_rescaling(self.mean_x, self.std_x)
        x = (x - self.mean_x) / self.std_x
        return x

    def rescale_at_index_x(self, x, index):
        x = (x - self.mean_x[index]) / self.std_x[index]
        return x

    def get_dataset(
        self,
        n_max: Union[None, int] = None,
        **kwargs,
    ):
        """
        :param use_y_rescaling:
        :param use_x_rescaling:
        :param n_max: maximal number of training data, set to None if one wants to use all of it...
        :param kwargs:
        :return: x, y
        """
        x, y = self.load_dataset()
        if n_max is not None:
            self.n_max = n_max
            x = x[0:n_max]
            y = y[0:n_max]
        if self.use_x_rescaling:
            self.mean_x, self.std_x = self.get_rescaling_factor(x)
            x = self.rescale_x(x)
        if self.use_y_rescaling:
            y = self.rescale_y(y)
        return x, y

    def rescale_theta(self, theta: Union[float, torch.Tensor]):
        if self.use_y_rescaling:
            theta = self.std_y.item() * theta + self.mean_y.item()
        return theta

    def rescale_theta_var(self, theta_var: float):
        if self.use_y_rescaling:
            theta_var = self.std_y**2 * theta_var
        return theta_var

    def load_dataset(self):
        if not self.dataset_exists():
            raise FileNotFoundError("Dataset does not exist and needs to be generated")
        x, y = self.load_dataset_from_disk()
        return x, y

    def dataset_exists(self) -> bool:
        return os.path.exists(os.path.join(self.dataset_path, f"{self.name}.pt"))

    def load_dataset_from_disk(self):
        dataset = torch.load(os.path.join(self.dataset_path, f"{self.name}.pt"))
        return dataset["x"], dataset["y"]

    def save_data(self, x: torch.Tensor, y: torch.Tensor):
        dataset = {"y": y, "x": x}
        os.makedirs(self.dataset_path, exist_ok=True)
        torch.save(dataset, os.path.join(self.dataset_path, f"{self.name}.pt"))

    def get_theta_mc(self):
        _, y = self.get_dataset(self.n_max)
        if self.n_max is not None:
            return torch.mean(y[0: self.n_max])
        return torch.mean(y)

    def get_integration_boundaries(self):
        a = self.a
        b = self.b
        if self.use_x_rescaling and self.a is not None:
            a = self.rescale_at_index_x(self.a, self.index_of_bounds)
        if self.use_x_rescaling and self.b is not None:
            b = self.rescale_at_index_x(self.b, self.index_of_bounds)
        return a, b


class DataClass(GenericDataClass):
    visualization_dim: int = 0
    NAME = ""
    dim = 1

    def __init__(
        self,
        use_x_rescaling: bool,
        use_y_rescaling: bool,
        dataset_size: int,
        version: int,
        dim: int,
        *args,
        **kwargs
    ):
        super(DataClass, self).__init__(use_x_rescaling, use_y_rescaling, args, kwargs)
        self.dim = dim
        self.version = version
        self.pdf = StandardNormalPDF(dim=dim)
        self.emukit_pdf = standard_normal(dim=dim)
        self.dataset_size = dataset_size
        self.dataset_path = str(Path(__file__).parent.parent / "datasets")
        self.base_size = 1000000
        self.name = f"{self.NAME}_dim_{self.dim}_v_{self.version}"

    def load_dataset(self):
        if not self.dataset_exists():
            self.generate_and_save_dataset()
        x, y = self.load_dataset_from_disk()
        return x, y

    def generate_and_save_dataset(self):
        y, x = self.generate_data()
        self.save_data(x, y)

    def generate_data(self):
        np.random.seed(self.version)
        torch.manual_seed(self.version)
        x = self.pdf.sample(self.base_size)
        y = self.f(x)
        np.random.seed(0)
        torch.manual_seed(0)
        return y, x
