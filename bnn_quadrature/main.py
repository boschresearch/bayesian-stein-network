import numpy as np
import torch

from bnn_quadrature.options import ModelEnum, DatasetEnum, Options
from bnn_quadrature.training import run_bsn, run_score, run_bq, run_scf
from bnn_quadrature.options.train_base import init_options

np.random.seed(0)
torch.manual_seed(0)

if __name__ == "__main__":
    options = Options
    opts = init_options(options)
    if opts.model_type == ModelEnum.bsn:
        if (
            opts.dataset_name == DatasetEnum.ode_4
        ):
            run_score(opts)
        else:
            run_bsn(opts)
    elif opts.model_type == ModelEnum.bq:
        if (
            opts.dataset_name == DatasetEnum.ode_4
        ):
            raise NotImplementedError(f"BQ not implemented for this dateset! Dataset: {opts.dataset_name}")
        run_bq(opts)
    elif opts.model_type == ModelEnum.scf:
        if (
            opts.dataset_name == DatasetEnum.ode_4 or
            opts.dataset_name == DatasetEnum.wind_farm_7d
        ):
            raise NotImplementedError(f"BQ not implemented for this dateset! Dataset: {opts.dataset_name}")
        run_scf(opts)
