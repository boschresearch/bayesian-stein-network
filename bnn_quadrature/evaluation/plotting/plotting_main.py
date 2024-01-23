from enum import Enum

import matplotlib.pyplot as plt
from tueplots import figsizes, axes, fontsizes

paper_path = "C:\\Users\\otk2rng\\Documents\\Submissions\\PaperFX\\draft\\fig"
PAGE_WIDTH = 6.75
HALF_PAGE_WIDTH = 3.25


def init_figure_half_column():
    plt.rcParams.update(axes.legend())
    plt.rcParams.update(katha_fontsizes)
    plt.rcParams.update(figsizes.icml2022_half())
    fig_size = figsizes.icml2022_half()
    fig_size["figure.figsize"] = (HALF_PAGE_WIDTH, 2.)
    plt.rcParams.update(fig_size)
    fig, ax = plt.subplots()
    return fig, ax

def init_figure_half_half_column():
    plt.rcParams.update(axes.legend())
    plt.rcParams.update(katha_fontsizes)
    plt.rcParams.update(figsizes.icml2022_half())
    fig_size = figsizes.icml2022_half()
    fig_size["figure.figsize"] = (HALF_PAGE_WIDTH/2, 1.8)
    plt.rcParams.update(fig_size)
    fig, ax = plt.subplots()
    return fig, ax


def init_figure_half_column2():
    plt.rcParams.update(axes.legend())
    plt.rcParams.update(katha_fontsizes)
    plt.rcParams.update(figsizes.icml2022_half())
    fig_size = figsizes.icml2022_half()
    fig_size["figure.figsize"] = (HALF_PAGE_WIDTH, 1.8)
    plt.rcParams.update(fig_size)
    fig, ax = plt.subplots(1, 2)
    return fig, ax

def init_figure_half_column22():
    plt.rcParams.update(axes.legend())
    plt.rcParams.update(katha_fontsizes)
    plt.rcParams.update(figsizes.icml2022_half())
    fig_size = figsizes.icml2022_half()
    fig_size["figure.figsize"] = (HALF_PAGE_WIDTH, 3.8)
    plt.rcParams.update(fig_size)
    fig, ax = plt.subplots(2, 2)
    return fig, ax

def init_figure_half_beamer():
    plt.rcParams.update(axes.legend())
    plt.rcParams.update(fontsizes.neurips2021())
    fig_size = figsizes.icml2022_half()
    fig_size["figure.figsize"] = (2., 1.3)
    plt.rcParams.update(fig_size)
    fig, ax = plt.subplots()
    return fig, ax


def init_figure_half_square():
    fig, ax = plt.subplots()
    fig.set_size_inches(1.8, 1.8)
    return fig, ax


katha_fontsizes = {
    "font.size": 7,
    "axes.labelsize": 7,
    "legend.fontsize": 5,
    "xtick.labelsize": 5,
    "ytick.labelsize": 5,
    "axes.titlesize": 9
}


def figure_full_4():
    plt.rcParams.update(axes.legend())
    plt.rcParams.update(katha_fontsizes)
    fig_size = figsizes.icml2022_half()
    fig_size["figure.figsize"] = (PAGE_WIDTH, 1.7)
    plt.rcParams.update(fig_size)
    fig, ax = plt.subplots(1, 4)
    return fig, ax


def figure_full_3():
    plt.rcParams.update(axes.legend())
    plt.rcParams.update(katha_fontsizes)
    fig_size = figsizes.icml2022_half()
    fig_size["figure.figsize"] = (PAGE_WIDTH, 1.7)
    plt.rcParams.update(fig_size)
    fig, ax = plt.subplots(1, 3)
    return fig, ax


def figure_full_3_2():
    plt.rcParams.update(axes.legend())
    plt.rcParams.update(katha_fontsizes)
    fig_size = figsizes.icml2022_half()
    fig_size["figure.figsize"] = (PAGE_WIDTH, 3.5)
    plt.rcParams.update(fig_size)
    fig, ax = plt.subplots(2, 3)
    return fig, ax


def figure_full_2_2():
    plt.rcParams.update(axes.legend())
    plt.rcParams.update(katha_fontsizes)
    fig_size = figsizes.icml2022_half()
    fig_size["figure.figsize"] = (PAGE_WIDTH, 3.5)
    plt.rcParams.update(fig_size)
    fig, ax = plt.subplots(2, 2)
    return fig, ax


def figure_full_2():
    plt.rcParams.update(axes.legend())
    plt.rcParams.update(katha_fontsizes)
    fig_size = figsizes.icml2022_half()
    fig_size["figure.figsize"] = (PAGE_WIDTH, 1.7)
    plt.rcParams.update(fig_size)
    fig, ax = plt.subplots(1, 2)
    return fig, ax


class MethodEnum(str, Enum):
    sgd = "sgd"
    bq = "bq"
    gp = "gp"
    nn = "nn"
    nn_16_0 = "nn_16_0"
    nn_16_2 = "nn_16_2"
    nn_16_4 = "nn_16_4"
    nn_32_0 = "nn_32_0"
    nn_32_2 = "nn_32_2"
    nn_32_4 = "nn_32_4"
    nn_64_0 = "nn_64_0"
    nn_64_2 = "nn_64_2"
    nn_64_4 = "nn_64_4"
    qmc = "qmc"
    qmc2 = "qmc2"
    cube = "cube"
    mc = "mc"
    mala = "mala"
    o_adam = "o_adam"
    o_sgd = "o_sgd"
    celu = "celu"
    gauss = "gauss"
    gelu = "gelu"
    tanh = "tanh"
    sigmoid = "sigmoid"
    silu = "silu"
    tanhshrink = "tanhshrink"
    none = "none"
    std = "std"
    max = "max"
    c500 = "c500"


settings = {
    "lw": 0.7,
    "markeredgecolor": "k",
    "ms": 3,
    "markeredgewidth": 0.5,
}

settings2 = {
    "lw": 0.7,
    "markeredgecolor": "k",
    "ms": 3,
    "markeredgewidth": 0.5,
}
