import os

from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from tueplots import axes, fontsizes, figsizes

from bnn_quadrature.evaluation.plotting.plotting_main import HALF_PAGE_WIDTH, paper_path, katha_fontsizes
from bnn_quadrature.evaluation.util.folder import colors

plt.rcParams.update(axes.legend())
plt.rcParams.update(katha_fontsizes)
plt.rcParams['axes.linewidth'] = 1.5
fig_size = figsizes.icml2022_half()
fig_size["figure.figsize"] = (HALF_PAGE_WIDTH, 1.5)
plt.rcParams.update(fig_size)
fig, ax = plt.subplots()
ax.spines[["top", "right"]].set_visible(False)

# Draw arrows (as black triangles: ">k"/"^k") at the end of the axes.  In each
# case, one of the coordinates (0) is a data coordinate (i.e., y = 0 or x = 0,
# respectively) and the other one (1) is an axes coordinate (i.e., at the very
# right/top of the axes).  Also, disable clipping (clip_on=False) as the marker
# actually spills out of the axes.

ellipse = Ellipse((1.2, 1.3),
                  width=1.5,
                  height=1.4,
                  facecolor=f"#{colors[5]}",
                  alpha=0.2)
ax.add_patch(ellipse)
ellipse = Ellipse((2.7, 2.7),
                  width=2,
                  height=2.2,
                  facecolor=f"#{colors[6]}",
                  alpha=0.2)
ax.add_patch(ellipse)
ax.plot(0.7, 1., "o", ms=3, color=f"#{colors[0]}")
ax.plot(1, 1.5, "o", ms=3, color=f"#{colors[0]}")
ax.plot(1.3, 1., "o", ms=3, color=f"#{colors[0]}")
ax.plot(1.9, 2.5, "o", ms=3, color=f"#{colors[1]}")
ax.plot(2.9, 3.4, "o", ms=3, color=f"#{colors[1]}")
ax.text(.8, 0.9, "MC", color=f"#{colors[0]}")
ax.text(1.4, 0.9, "MCMC", color=f"#{colors[0]}")
ax.text(1.1, 1.4, "QMC", color=f"#{colors[0]}")
ax.text(2., 2., "Bayesian\nStein\nnetwork", color=f"#{colors[1]}")
ax.text(3., 3.3, "BQ", color=f"#{colors[1]}")
ax.text(2.0, 3.5, "PN Methods", color=f"#{colors[6]}", fontweight="bold")
ax.text(0.7, 1.9, "MC Methods", color=f"#{colors[5]}", fontweight="bold")
ax.set_xlabel("Computational cost")
ax.set_ylabel("Prior Information\nEncodable in Method")
ax.set_xlim([0.5, 4])
ax.set_ylim([0.5, 4])
ax.plot(1., 0.5, ">k", transform=ax.get_yaxis_transform(), clip_on=False, ms=2.5)
ax.plot(0.5, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False, ms=2.5)
ax.set_xticks([])
ax.set_yticks([])
# ax.legend()
fig.savefig(os.path.join(paper_path, "sketch.png"), dpi=500)
fig.savefig(os.path.join(paper_path, "sketch.pdf"), dpi=500)
