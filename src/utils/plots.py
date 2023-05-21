import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import wandb
from matplotlib.ticker import FormatStrFormatter


def sweep_plot(
    eval_df,
    name,
    x="n",
    y="value",
    col="epsilon",
    row="loss",
    hue="method",
    style="n_feats",
    format="jpeg",
    show=False,
    save=True,
    wandb_log=True,
    share_y=False,
    share_x=True,
    palette="colorblind",
    bbox_to_anchor=None,
    dashes=None,
    col_name=r"$\epsilon$" + "={col_name}",
    row_name=None,
    legend_ncol=3,
    legend_font=19,
    set_supxlabel=True,
    aspect=1.2,
    scale=2,
    return_fig=False,
    figsize=(11.7, 8.27),
    y_max=2,
    no_legend=False,
    set_left_low_xlabel=False,
    filter_col=True,
):
    assert show + save < 2, "Can only show or save, not both"

    sns.set(
        font_scale=scale,
        style="white",
        rc={
            "font.family": "Times New Roman",
            "figure.constrained_layout.use": True,
            "figure.figsize": figsize,
        },
    )

    g = sns.relplot(
        data=eval_df,
        x=x,
        y=y,
        hue=hue,
        col=col,
        row=row,
        style=style,
        kind="line",
        errorbar="se",
        linewidth=3,
        marker="o",
        facet_kws={"sharey": share_y, "sharex": share_x, "margin_titles": True},
        palette=sns.color_palette(palette),
        dashes=dashes,
        aspect=aspect,
    )

    # set y labels
    if row_name is not None:
        for i, row_label in enumerate(g.row_names):
            g.facet_axis(i, 0).set_ylabel(row_name(row_label))
    else:
        for i, row_label in enumerate(g.row_names):
            g.facet_axis(i, 0).set_ylabel(row_label)

    g.set_xlabels("", clear_inner=False)

    g.set_titles(col_template=col_name, row_template="")
    if set_supxlabel:
        g.fig.supxlabel(x)
    elif set_left_low_xlabel:
        g.facet_axis(-1, 0).set_xlabel(x)

    g.fig.tight_layout()

    if len(g.row_names) == 1:
        ticks = g.facet_axis(-1, 0).get_xticks()[::2]
    for i, _col_label in enumerate(g.col_names):
        # g.facet_axis(-1, i).xaxis.get_major_locator().set_params(integer=True)
        g.facet_axis(-1, i).yaxis.set_major_formatter(
            FormatStrFormatter("%.1f")
        )
        g.facet_axis(0, i).yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        if len(g.row_names) > 1:
            g.facet_axis(0, i).set_xticks([])
            g.facet_axis(-1, i).set_xticks(
                g.facet_axis(-1, i).get_xticks()[::2]
            )
        elif filter_col:
            g.facet_axis(-1, i).set_xticks(ticks)
        g.facet_axis(-1, i).set_ylim(
            top=min(y_max, g.facet_axis(-1, i).get_ylim()[1])
        )

    if no_legend:
        try:
            g._legend.remove()
        except:
            pass
    else:
        # get the legend object
        leg = g._legend

        # change the line width for the legend
        for line in leg.get_lines():
            line.set_linewidth(4.0)

        sns.move_legend(
            g,
            "lower right",
            title=None,
            ncol=legend_ncol,
            fontsize=legend_font / 1.5 * scale,
            bbox_to_anchor=bbox_to_anchor,
            columnspacing=0.8,
        )

    plt.subplots_adjust(
        # left=0.1,
        # bottom=0.1,
        # right=0.9,
        # top=0.9,
        wspace=3,
        hspace=0.4,
    )

    if return_fig:
        return g

    if show:
        plt.show()

    if save:
        plt.savefig(f"{name}.{format}", bbox_inches="tight", format=format)
        plt.close()

    if wandb_log:
        wandb.log({"plot": wandb.Image(f"{name}.{format}")})


def plot_data(data):
    X, y = data
    sns.regplot(X, y, fit_reg=False)
    plt.scatter(X, y)


def abline(slope, intercept, ax):
    """Plot a line from slope and intercept"""
    x_vals = np.array(ax.get_xlim())
    y_vals = intercept + slope * x_vals
    ax.plot(x_vals, y_vals, "--", c="red")


def reg_plot(ds, theta=None, ax=None, show=True):
    sns.regplot(x=ds.x[:, -1], y=ds.y, fit_reg=True, ax=ax, ci=None)
    theta = ds.theta_gen["theta"] if theta is None else theta
    abline(
        theta[-1],
        theta[0] if len(theta) > 1 else 0,
        plt.gca() if ax is None else ax,
    )
    if show:
        plt.show()


def distringuish_reg_plot(d0, d1, attack_idx, theta_fitted):
    d = [d0, d1][attack_idx]
    df = [d0, d1][1 - attack_idx]
    _fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True, sharex=True)
    reg_plot(d, theta_fitted, ax=axs[0], show=False)
    reg_plot(df, theta_fitted, ax=axs[1], show=True)
