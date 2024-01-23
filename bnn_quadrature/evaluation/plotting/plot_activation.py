import os

from bnn_quadrature.evaluation.plotting.plot_loss import PlotLossComparison
from bnn_quadrature.evaluation.plotting.plot_comparison import PlotComparison
from bnn_quadrature.evaluation.plotting.plot_runtime_vs_rmse import PlotRuntimeError
from bnn_quadrature.evaluation.plotting.plotting_main import figure_full_3


def plot_act(name, fig_path):
    fig, ax = figure_full_3()
    data_name = name
    plot = PlotLossComparison(data_name)
    plot.plot(ax[0], fig)
    plot = PlotComparison(data_name)
    plot.plot(ax[1], fig)
    plot = PlotRuntimeError(data_name)
    plot.plot(ax[2], fig)
    fig.savefig(os.path.join(fig_path, f"{name}.png"), dpi=500)
    fig.savefig(os.path.join(fig_path, f"{name}.pdf"))
    fig.clf()
