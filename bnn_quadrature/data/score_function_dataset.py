import os
import warnings
from pathlib import Path
from typing import Union, Tuple

import pandas
import torch


class ScoreDataClass:
    visualization_dim: int
    dim: int

    def __init__(
        self,
        index: int,
        version: int,
        use_x_rescaling: bool = False,
        use_y_rescaling: bool = False,
        *args,
        **kwargs,
    ):
        super(ScoreDataClass, self).__init__()
        self.index = index
        self.a = torch.zeros(self.dim)
        self.b = None
        self.version = version
        self.index_of_bounds = torch.arange(0, self.dim, 1)
        warnings.warn(f"Not using {args}, {kwargs}")
        self.dataset_path = str(Path(__file__).parent.parent / "datasets")
        self.name = f"{self.NAME}"
        self.use_x_rescaling = use_x_rescaling
        self.use_y_rescaling = use_y_rescaling
        self.mean_x = None
        self.std_x = None
        self.mean_y = None
        self.std_y = None
        self.n_max = None

    def get_rescaling_factor(self, x) -> Union[Tuple[torch.Tensor, torch.Tensor], None]:
        if self.use_x_rescaling:
            mean = torch.mean(x, dim=0).squeeze(0)
            std = torch.std(x, dim=0).squeeze(0)
            return mean, std

    def f(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def rescale_y(self, y):
        y_rescaling_factor = torch.max(y) - torch.min(y)
        y_min = torch.min(y)
        y = (y - y_min) / y_rescaling_factor
        self.mean_y = y_min
        self.std_y = y_rescaling_factor
        return y

    def rescale_x(self, x):
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
        :return: x, y, x_test of shape
        """
        x, y, scores = self.load_dataset()
        if self.use_x_rescaling:
            self.mean_x, self.std_x = self.get_rescaling_factor(x)
            x = self.rescale_x(x)
        if self.use_y_rescaling:
            y = self.rescale_y(y)
        if n_max is not None:
            self.n_max = n_max
            x = x[0:n_max]
            y = y[0:n_max]
            scores = scores[0:n_max]
        x = torch.cat((x, scores), dim=-1)
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
        x, y, scores = self.load_dataset_from_disk()
        return x, y, scores

    def dataset_exists(self) -> bool:
        return os.path.exists(os.path.join(self.dataset_path, f"{self.name}_x_v_{self.version}.csv"))

    def load_dataset_from_disk(self):
        x = pandas.read_csv(os.path.join(self.dataset_path, f"{self.name}_x_v_{self.version}.csv")).values
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
        scores = pandas.read_csv(os.path.join(self.dataset_path, f"{self.name}_scores_v_{self.version}.csv")).values
        scores = torch.tensor(scores, dtype=torch.float32).unsqueeze(1)
        y = x[:, :, self.index]
        return x, y, scores

    def get_theta_mc(self):
        _, y = self.get_dataset(self.n_max)
        if self.n_max is not None:
            return torch.mean(y[0: self.n_max])
        return torch.mean(y)

    def get_integration_boundaries(self):
        return self.a, self.b


class ODEBaseDataset(ScoreDataClass):
    def true_solution(self):
        solution = torch.cat((torch.tensor([1, 3, 1.]), torch.ones(self.dim-4), torch.tensor([0.5])))
        return solution[self.index]


class ODE4Dataset(ODEBaseDataset):
    NAME = "goodwin_4"
    dim = 4


