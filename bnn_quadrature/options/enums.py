import enum
from enum import Enum


class MetaEnum(enum.EnumMeta):
    def __contains__(cls, item):
        return item in cls.__members__.values()


class ModelEnum(str, Enum):
    bsn = "bsn"
    bq = "bq"
    scf = "scf"


class DeviceEnum(str, Enum):
    cuda = "cuda"
    cpu = "cpu"


class TransformationEnum(str, Enum):
    none = "none"
    const = "const"


class DatasetEnum(str, Enum):
    genz_continuous_integral = "genz_continuous_integral"
    genz_continuous_qmc = "genz_continuous_qmc"
    genz_corner_peak = "genz_corner_peak"
    genz_discontinuous = "genz_discontinuous"
    genz_gaussian = "genz_gaussian"
    genz_oscillatory = "genz_oscillatory"
    genz_product = "genz_product"
    wind_farm_7d = "wind_farm_7d"
    ode_4 = "ode_4"


class PytorchSolverEnum(str, Enum, metaclass=MetaEnum):
    adam = "adam"
    sgd = "sgd"
    pt_l_bfgs = "pt_l_bfgs"
    hessian_free = "hessian_free"


class ActivationEnum(str, Enum):
    tanh = "tanh"
    relu = "relu"
    celu = "celu"
    sigmoid = "sigmoid"
    gelu = "gelu"
    silu = "silu"
    tanhshrink = "tanhshrink"
    gauss = "gauss"
