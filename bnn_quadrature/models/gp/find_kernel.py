from enum import Enum

from gpytorch.kernels import RBFKernel, ScaleKernel


class KernelEnum(Enum):
    rbf_kernel = "rbf_kernel"


def find_kernel(kernel: KernelEnum) -> ScaleKernel:
    if kernel == KernelEnum.rbf_kernel:
        return ScaleKernel(RBFKernel())
    else:
        raise NotImplementedError(
            f"The Kernel {kernel} is not implemented. Check 'KernelEnum' for all implemented Kernels"
        )
