def plot_my_error(ax, niter, std_time, time, lolims=True, uplims=False):
    (_, caps, _) = ax.errorbar(
        niter,
        time,
        yerr=std_time,
        ls="none",
        # solid_capstyle="projecting",
        capsize=1.0,
        lolims=lolims,
        uplims=uplims,
        color="black",
        lw=0.5,
    )

    for cap in caps:
        cap.set_markeredgewidth(0.5)
