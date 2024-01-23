from typing import Type, Union

from bnn_quadrature.data.dataset_base import GenericDataClass
from bnn_quadrature.data.genz_data import (
    GenzCornerPeak,
    DiscontinuousGenz,
    GenzGaussian,
    GenzOscillatory,
    GenzProductPeak,
    ContinuousGenz,
    GenzContinuousQMCIntegral,
)
from bnn_quadrature.data.score_function_dataset import ODE4Dataset, ScoreDataClass
from bnn_quadrature.data.wind_farm.wind_farm_dataset_7d import WindFarmDataset7D
from bnn_quadrature.options.enums import DatasetEnum


def find_dataset(
    dataset_name: str,
) -> Union[Type[GenericDataClass], Type[ScoreDataClass]]:
    if dataset_name == DatasetEnum.genz_continuous_integral:
        return ContinuousGenz
    elif dataset_name == DatasetEnum.genz_continuous_qmc:
        return GenzContinuousQMCIntegral
    elif dataset_name == DatasetEnum.genz_corner_peak:
        return GenzCornerPeak
    elif dataset_name == DatasetEnum.genz_discontinuous:
        return DiscontinuousGenz
    elif dataset_name == DatasetEnum.genz_gaussian:
        return GenzGaussian
    elif dataset_name == DatasetEnum.genz_oscillatory:
        return GenzOscillatory
    elif dataset_name == DatasetEnum.genz_product:
        return GenzProductPeak
    elif dataset_name == DatasetEnum.wind_farm_7d:
        return WindFarmDataset7D
    elif dataset_name == DatasetEnum.ode_4:
        return ODE4Dataset
    else:
        raise KeyError(f"The dataset {dataset_name} is not implemented!!!")
