import os
from typing import List

from bnn_quadrature.evaluation.plotting.plot_calibration import PlotCalibration
from bnn_quadrature.evaluation.plotting.plot_comparison import PlotComparison
from bnn_quadrature.evaluation.plotting.plot_loss import PlotLossComparison
from bnn_quadrature.evaluation.plotting.plot_run_time import PlotRunTime
from bnn_quadrature.evaluation.plotting.plot_runtime_vs_rmse import PlotRuntimeError
from bnn_quadrature.evaluation.plotting.plot_uncertainty import PlotUncertainty
from bnn_quadrature.evaluation.plotting.plotting_main import figure_full_4
from bnn_quadrature.evaluation.util.folder import Folder


def generate_plots(
    dim,
    folder_list: List[Folder],
    name,
    true_value,
    plot_uncertainty=True,
):
    plot = PlotRuntimeError(name)
    plot.generate_data(
        dim, folder_list, true_value, name=name
        )
    plot.figure()
    plot = PlotRunTime(name)
    plot.generate_data(dim, folder_list, name=name)
    plot.figure()
    plot = PlotComparison(name)
    plot.generate_data(
        dim, folder_list, true_value, name=name
    )
    plot.figure()
    if plot_uncertainty:
        plot = PlotUncertainty(name)
        plot.generate_data(
            dim,
            folder_list,
            true_value,
            name=name,
        )
        plot.figure()
        plot = PlotCalibration(name)
        plot.generate_data(
            dim,
            folder_list,
            true_value,
            name=name,
        )
        plot.figure()


def generate_loss_plots(
    folder_list: List[Folder],
    name,
):
    plot = PlotLossComparison(name)
    plot.generate_data(
        folder_list,
        name=name
    )
    plot.figure()


def plot_figure(name, fig_path):
    fig, ax = figure_full_4()
    plot = PlotComparison(name)
    plot.plot(ax[0], fig)
    plot = PlotRunTime(name)
    plot.plot(ax[1], fig)
    plot = PlotRuntimeError(name)
    plot.plot(ax[2], fig)
    plot = PlotCalibration(name)
    plot.plot(ax[3], fig)
    fig.savefig(os.path.join(fig_path, f"{name}.png"), dpi=500)
    fig.savefig(os.path.join(fig_path, f"{name}.pdf"))
    fig.clf()


def plot_figure_uncertainty(name, fig_path):
    fig, ax = figure_full_4()
    plot = PlotComparison(name)
    plot.plot(ax[0], fig)
    plot = PlotRunTime(name)
    plot.plot(ax[1], fig)
    plot = PlotRuntimeError(name)
    plot.plot(ax[2], fig)
    plot = PlotUncertainty(name)
    plot.plot(ax[3], fig)
    fig.savefig(os.path.join(fig_path, f"{name}.png"), dpi=500)
    fig.savefig(os.path.join(fig_path, f"{name}.pdf"))
    fig.clf()

