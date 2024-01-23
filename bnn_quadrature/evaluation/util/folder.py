from typing import Union

from tueplots.constants.color import palettes

from bnn_quadrature.evaluation.plotting.plotting_main import MethodEnum

colors = palettes.muted


class Folder:
    name: Union[MethodEnum, str]

    def __init__(self, folder_name):
        self.folder_name: Union[str, None] = folder_name

    def get_settings(self):
        ...


class FolderNN(Folder):
    name = MethodEnum.nn

    def get_settings(self):
        return {"marker": "v", "label": "BSN", "color": f"#{colors[0]}"}


class FolderMC(Folder):
    name = MethodEnum.mc

    def get_settings(self):
        return {"marker": "o", "label": "MC", "color": f"#{colors[1]}"}


class FolderSGD(Folder):
    name = MethodEnum.sgd

    def get_settings(self):
        return {"marker": "v", "label": "Stein-NN", "color": f"#{colors[0]}", "alpha": 0.3}


class FolderGP(Folder):
    name = MethodEnum.gp

    def get_settings(self):
        return {"marker": "s", "label": "Stein-CF", "color": f"#{colors[3]}"}


class FolderBQ(Folder):
    name = MethodEnum.bq

    def get_settings(self):
        return {"marker": "X", "label": "BQ", "color": f"#{colors[2]}"}


def find_folder(name: Union[MethodEnum, str]) -> Folder:
    if name == MethodEnum.nn:
        return FolderNN(None)
    elif name == MethodEnum.sgd:
        return FolderSGD(None)
    elif name == MethodEnum.gp:
        return FolderGP(None)
    elif name == MethodEnum.bq:
        return FolderBQ(None)
    elif name == MethodEnum.mc:
        return FolderMC(None)
    else:
        raise NotImplementedError(f"Method {name} not implemented!")
