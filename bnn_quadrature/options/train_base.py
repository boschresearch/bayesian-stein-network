import argparse
from typing import Dict, Type

import yaml

from bnn_quadrature.options.options import Options


def train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="Name of the experiment to run.")
    parser.add_argument("--output_dir", type=str, help="Name of the experiment to run.")
    parser.add_argument(
        "--options_file", type=str, help="Name of the experiment to run."
    )
    return parser


def load_experiment_dict(options_file: str) -> Dict:
    exp_dir = options_file
    with open(exp_dir, "r") as yaml_in:
        opts_dict = yaml.safe_load(yaml_in)
    assert isinstance(opts_dict, dict)
    return opts_dict


def init_options(
    options: Type[Options]
) -> Options:
    parser = train_parser()
    args = parser.parse_args()
    opts = options()
    if args.options_file:
        opts_dict = load_experiment_dict(args.options_file)
        opts = options(**opts_dict).parse_obj(opts_dict)
    if args.output_dir:
        opts.output_dir = args.output_dir
        opts.name = args.name
    opts.initialize_setup()
    return opts
