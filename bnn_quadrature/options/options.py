import os
import warnings
from pathlib import Path
from typing import Dict, Any

import torch
from pydantic import BaseModel

from bnn_quadrature.cluster_functionalities.yaml_service import save_yaml
from bnn_quadrature.models.gp.find_kernel import KernelEnum
from bnn_quadrature.options.device import my_device
from bnn_quadrature.options.enums import (
    DatasetEnum,
    PytorchSolverEnum, TransformationEnum, DeviceEnum, ActivationEnum, ModelEnum,
)


class Options(BaseModel):
    name: str = "bnn_quadrature"

    model_type: ModelEnum = ModelEnum.bsn

    dataset_name: DatasetEnum = DatasetEnum.genz_continuous_integral
    device: DeviceEnum = DeviceEnum.cpu
    version: int = 0
    dataset_size: int = 80
    set_batch_to_dataset_size: bool = True
    batch_size: int = 50
    weight_decay: float = 1e-8
    mean_dim: int = 2   # only relevant for the ODE based datasets
    dim = 1

    const: str = "std"

    hidden_dim: int = 32
    act: ActivationEnum = ActivationEnum.celu
    num_layers: int = 2
    start_theta_0: float = 0.0

    solver: PytorchSolverEnum = PytorchSolverEnum.pt_l_bfgs
    max_iter: int = 1001
    lr: float = 0.01

    """
    Settings specific for SCF model 
    """
    kernel: KernelEnum = KernelEnum.rbf_kernel
    eval_interval: int = 50
    noise_key = "likelihood.noise_covar.noise"
    lengthscale_key = "gp.sk.kernel.base_kernel.lengthscale"
    lengthscale: float = 0.3
    noise: float = 1e-4
    hps: Dict[str, Any] = {
        noise_key: torch.tensor(noise),
        lengthscale_key: torch.tensor(lengthscale),
    }

    transformation: TransformationEnum = TransformationEnum.none

    output_dir: str = str(Path(__file__).parent.parent.parent / "results")
    results_folder: str = None
    do_run_evaluation: bool = True
    do_train_model: bool = True
    do_fit_laplace: bool = True
    do_eval_uncertainty: bool = True

    """
    Options specific for the wind farm model_type
    """
    use_x_rescaling: bool = True
    use_y_rescaling: bool = True

    class Config:
        use_enum_values = True

    def initialize_setup(self):
        self.results_folder = os.path.join(self.output_dir, self.name)
        os.makedirs(self.results_folder, exist_ok=True)
        if self.set_batch_to_dataset_size:
            self.batch_size = self.dataset_size
        if self.solver == PytorchSolverEnum.pt_l_bfgs:
            if not self.batch_size == self.dataset_size:
                raise ValueError(
                    f"Batch size and dataset size have to be equal for solver {self.solver}"
                )
        my_device.device = self.device
        if self.batch_size > self.dataset_size:
            warnings.warn("Batch size larger than dataset size - setting batch size to dataset size")
            self.batch_size = self.dataset_size

        save_yaml(self.dict(), os.path.join(self.results_folder, "opts.yaml"))
