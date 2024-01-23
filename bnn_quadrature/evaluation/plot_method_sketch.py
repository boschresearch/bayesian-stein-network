import os
from pathlib import Path

import torch
from matplotlib import pyplot as plt
from tueplots import axes, fontsizes, figsizes

from bnn_quadrature.data.genz_data.genz_continuous_integral import ContinuousGenz
from bnn_quadrature.data.pdf.pdf import PDF, NormalPDF

# plot f
from bnn_quadrature.evaluation.plotting.plotting_main import paper_path
from bnn_quadrature.evaluation.util.folder import colors

plt.rcParams.update(axes.legend())
plt.rcParams.update(fontsizes.neurips2021())
plt.rcParams['axes.linewidth'] = 1.5


def plot_f():
    data = ContinuousGenz(
        pdf=PDF,
        dim=1,
        use_y_rescaling=False,
        use_x_rescaling=False,
        dataset_size=20,
        test_dataset_size=1,
        version=0,
    )
    pdf = NormalPDF()
    x, f, _ = data.get_dataset(10)
    x_test = torch.linspace(-2, 2, 100).unsqueeze(1).unsqueeze(1)
    pdf_x = 0.1 * pdf.grad_log(x).flatten().detach() + 0.8
    pdf_x_test = 0.1 * pdf.grad_log(x_test).flatten().detach() + 0.8
    x = x.flatten().detach()
    f = f.flatten()
    f_x_test = data.f(x_test).flatten()
    x_test = x_test.flatten().detach()
    fig_size = figsizes.icml2022_half()
    fig_size["figure.figsize"] = (1.2, 1.1)
    plt.rcParams.update(fig_size)
    fig, ax = plt.subplots()
    ax.plot(x_test, f_x_test, alpha=0.5, lw=1.5, color=f"#{colors[0]}")
    ax.plot(x_test, pdf_x_test, alpha=0.5, lw=1.5, color=f"#{colors[1]}")
    ax.text(0.5, 0.9, "$f(x_i)$", color=f"#{colors[0]}")
    ax.text(-0.4, .55, "$\\nabla \log \pi(x_i)$", color=f"#{colors[1]}")
    ax.plot(x, f, "o", ms=3., label="$f(x_i)$", color=f"#{colors[0]}")
    ax.plot(x, pdf_x, "o", ms=3., label="$\\nabla \log \pi(x_i)$", color=f"#{colors[1]}")
    ax.set_xlabel("x")
    ax.set_xlim([-2, 2])
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.legend()
    fig.savefig(os.path.join(paper_path, "f.png"), dpi=500)
    fig.savefig(os.path.join(paper_path, "f.pdf"), dpi=500)


plot_f()
def plot_fit():
    global data, pdf, x, model
    file_dir = Path(__file__).parent / "data" / "genz_1_10" / "f.pt"
    data = ContinuousGenz(
        pdf=PDF,
        dim=1,
        use_y_rescaling=True,
        use_x_rescaling=True,
        dataset_size=20,
        test_dataset_size=1,
        version=0,
    )
    pdf = NormalPDF()
    x, f, _ = data.get_dataset(10)
    x = x.flatten()
    f = f.flatten()
    model = torch.load(file_dir)
    model_x = model["x_test"].flatten()
    model_f = model["model_mean"].flatten()
    fig_size = figsizes.icml2022_half()
    fig_size["figure.figsize"] = (1.2, 1.1)
    plt.rcParams.update(fig_size)
    fig, ax = plt.subplots()
    ax.plot(x, f, "o", ms=3., label="$f(x_i)$", color=f"#{colors[0]}")
    ax.plot(model_x, model_f, color=f"dimgrey")
    ax.text(0.65, 0.7, "$f(x_i)$", color=f"#{colors[0]}")
    ax.text(-0.65, 0.2, "$g_{\\theta_{\mathrm{MAP}}}(x)$", color=f"dimgray")
    ax.set_xlabel("x")
    ax.set_xlim([-2., 2.])
    ax.set_ylim([-.1, 1.])
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.legend()
    fig.savefig(os.path.join(paper_path, "g.png"), dpi=500)
    fig.savefig(os.path.join(paper_path, "g.pdf"), dpi=500)


plot_fit()
# plot uncertainty on theta
file_dir = Path(__file__).parent / "data" / "genz_1_10" / "theta_final.pt"
theta = torch.load(file_dir)
mean = theta["theta_mcmc_scaled"]
var = theta["var_theta_model_scaled"]
pdf = NormalPDF(mean, var)
x_min = mean - 10 * var
x_max = mean + 20 * var
x = torch.linspace(x_min, x_max, 100).unsqueeze(1).unsqueeze(1)
p = pdf(x)
x = x.flatten()
p = p.flatten()
fig_size = figsizes.icml2022_half()
fig_size["figure.figsize"] = (1.2, 1.1)
plt.rcParams.update(fig_size)
fig, ax = plt.subplots()
ax.plot(x, p, color=f"#{colors[6]}")
ax.fill_between(x, 0*p, p, color=f"#{colors[6]}", alpha=0.4)
ax.vlines(mean, 0, 100, color="black", ls="--")
ax.text(0.47, 1.3, "$\\theta_{0,\mathrm{MAP}}$", color=f"black")
ax.text(0.85, 0.7, "$p(\\theta_{0}\mid \mathcal{D})$", color=f"#{colors[6]}")
ax.set_xlabel("$\\theta_0$")
ax.set_xlim([x_min, x_max])
ax.set_ylim([0., 1.6])
ax.set_xticks([])
ax.set_yticks([])
fig.savefig(os.path.join(paper_path, "theta.png"), dpi=500)
fig.savefig(os.path.join(paper_path, "theta.pdf"), dpi=500)

print(theta)
