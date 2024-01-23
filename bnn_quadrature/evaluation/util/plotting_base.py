import os
from abc import ABC
from pathlib import Path

import torch

from bnn_quadrature.evaluation.plotting.plotting_main import init_figure_half_column


class PlottingClass(ABC):
    NAME = ""

    def __init__(self, data_name, data_dir=None):
        self.data_name = data_name
        if data_dir is None:
            self.data_dir = Path(__file__).parent / "data" / self.data_name
        else:
            self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.data_file = os.path.join(self.data_dir, f"{self.NAME}.pt")

    def figure(self):
        fig, ax = init_figure_half_column()
        self.plot(ax, fig)
        fig.savefig(os.path.join(str(self.data_dir), f"{self.NAME}.png"), dpi=500)
        fig.savefig(os.path.join(str(self.data_dir), f"{self.NAME}.pdf"))
        fig.clf()

    def plot(self, ax, fig, *args, **kwargs):
        ...

    def load(self):
        data_dict = torch.load(self.data_file)
        return data_dict

    def generate_data(self, *args, **kwargs):
        ...
