import torch
from pytest_mock import MockerFixture

from bnn_quadrature.main import run_bsn
from bnn_quadrature.options.enums import PytorchSolverEnum, DatasetEnum
from bnn_quadrature.options.options import Options

opts = Options()
opts.initialize_setup()
opts.max_iter = 2
opts.batch_size = 10
opts.dataset_size = 10
opts.dataset_name = DatasetEnum.genz_continuous_integral
opts.hidden_dim = 8
opts.num_layers = 0


def test_adam_solver(mocker: MockerFixture):
    solver = PytorchSolverEnum.adam
    opts.solver = solver
    spy_solver = mocker.spy(torch.optim, "Adam")
    run_bsn(opts)
    spy_solver.assert_called()


def test_sgd_solver(mocker: MockerFixture):
    solver = PytorchSolverEnum.sgd
    opts.solver = solver
    spy_solver = mocker.spy(torch.optim, "SGD")
    run_bsn(opts)
    spy_solver.assert_called()
