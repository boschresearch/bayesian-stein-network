from typing import Union

from bnn_quadrature.data.dataset_base import GenericDataClass
from bnn_quadrature.data.find_dataset import find_dataset
from bnn_quadrature.data.torch_dataset import get_data_loader
from bnn_quadrature.evaluation.evaluate import evaluate_training
from bnn_quadrature.models.nn.network import NeuralNetwork
from bnn_quadrature.models.nn.stein_model import SteinModel
from bnn_quadrature.options import my_device
from bnn_quadrature.options.options import Options
from bnn_quadrature.training.training_bsn import training


def run_bsn(opts: Options):
    integral = initialize_dataset(opts)
    x, y = integral.get_dataset(n_max=opts.dataset_size)
    x = x.to(my_device.device)
    y = y.to(my_device.device)
    data_loader = get_data_loader(x, y, opts.batch_size)
    model = initialize_model(opts, integral).to(my_device.device)
    if opts.do_train_model:
        training(
            model=model, data_loader=data_loader, opts=opts
        )
    if opts.do_run_evaluation:
        evaluate_training(
            model,
            x=x,
            y=y,
            opts=opts,
            integral=integral,
            true_solution=integral.true_solution(),
        )


def initialize_dataset(opts: Options) -> GenericDataClass:
    DatasetClass = find_dataset(opts.dataset_name)
    dataset = DatasetClass(
        use_x_rescaling=opts.use_x_rescaling,
        use_y_rescaling=opts.use_y_rescaling,
        dataset_size=opts.dataset_size,
        dim=opts.dim,
        version=opts.version
    )
    return dataset


def initialize_model(opts: Options, integral: GenericDataClass) -> SteinModel:
    network = NeuralNetwork(
        in_dim=integral.dim,
        hidden_dim=opts.hidden_dim,
        act=opts.act,
        out_dim=integral.dim,
        num_hidden_layers=opts.num_layers,
    )
    a, b = integral.get_integration_boundaries()
    return SteinModel(
        network=network,
        pdf=integral.pdf,
        start_theta_0=opts.start_theta_0,
        transformation=opts.transformation,
        a=a,
        b=b,
        index_of_bounds=integral.index_of_bounds
    )
