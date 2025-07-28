# %% ---------------------------------------------------------------------------------

import warnings

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn.objects as so

from grgg import make_paths, plotting

warnings.filterwarnings("ignore", message="last has more values", category=UserWarning)

# Paths
paths = make_paths()
paths.figures /= "spherical"
paths.figures.mkdir(parents=True, exist_ok=True)

simulation = joblib.load(paths.proc / "regimes-spherical.pkl")
params = simulation.params

# Plotting settings
mpl.rcParams.update(plotting.theme())

PLOT_HEIGHT = 2.5  # Height of the plots in inches
COLORS = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
MARKERS = mpl.rcParams["axes.prop_cycle"].by_key()["marker"]

# %%
# ====================================================================================
# SIMILARITY
# ====================================================================================

results = simulation.sim
simdata = results.drop(columns=["idx"]).groupby(["n", "k", "beta"]).mean().reset_index()

# %% Plot | Clustering ---------------------------------------------------------------

h = PLOT_HEIGHT
fig, axes = plt.subplots(
    ncols=(ncols := len(params.k)), figsize=(h * ncols, h), sharex=True, sharey=True
)

for ax, gdf in zip(axes.flat, simdata.groupby("k"), strict=True):
    k, gdf = gdf
    y = "clustering"
    (
        so.Plot(gdf, x="n", y=y, color="beta", marker="beta")
        .add(so.Line(), so.Dodge(by=y), legend=False)
        .add(so.Dot(), so.Dodge(by=y), legend=False)
        .scale(
            color=so.Nominal(values=COLORS),
            marker=so.Nominal(values=MARKERS),
        )
        .on(ax)
        .plot()
    )
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"k={k}")
    ax.set_ylim(top=1.0)

fig.supylabel(r"clustering", x=0.02)
fig.tight_layout()
fig.savefig(paths.figures / "regimes-sim-clust.pdf")

# %% Plot | Q-Clustering -------------------------------------------------------------

h = PLOT_HEIGHT
fig, axes = plt.subplots(
    ncols=(ncols := len(params.k)), figsize=(h * ncols, h), sharex=True, sharey=True
)

for ax, gdf in zip(axes.flat, simdata.groupby("k"), strict=True):
    k, gdf = gdf
    y = "qclustering"
    (
        so.Plot(gdf, x="n", y=y, color="beta", marker="beta")
        .add(so.Line(), so.Dodge(by=y), legend=False)
        .add(so.Dot(), so.Dodge(by=y), legend=False)
        .scale(
            color=so.Nominal(values=COLORS),
            marker=so.Nominal(values=MARKERS),
        )
        .on(ax)
        .plot()
    )
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"k={k}")
    ax.set_ylim(top=1.0)

fig.supylabel(r"$q$-clustering", x=0.02)
fig.tight_layout()
fig.savefig(paths.figures / "regimes-sim-qclust.pdf")

# %% Plot | Average Path Length ------------------------------------------------------

h = PLOT_HEIGHT
fig, axes = plt.subplots(
    ncols=(ncols := len(params.k)), figsize=(h * ncols, h), sharex=True, sharey=True
)

for ax, gdf in zip(axes.flat, simdata.groupby("k"), strict=True):
    k, gdf = gdf
    y = "average_path_length"
    (
        so.Plot(gdf, x="n", y=y, color="beta", marker="beta")
        .add(so.Line(), so.Dodge(by=y), legend=False)
        .add(so.Dot(), so.Dodge(by=y), legend=False)
        .scale(
            color=so.Nominal(values=COLORS),
            marker=so.Nominal(values=MARKERS),
        )
        .on(ax)
        .plot()
    )
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"k={k}")

fig.supylabel(r"Avg. path length (L)", x=0.02)
fig.tight_layout()
fig.savefig(paths.figures / "regimes-sim-paths.pdf")

# %% Shared legend -------------------------------------------------------------------

handles = [
    mpl.lines.Line2D(
        [],
        [],
        color=color,
        marker="o",
        linestyle="--",
        label=f"β'={beta:.2f}",
    )
    for color, beta in zip(COLORS, params.beta, strict=False)
]
fig, ax = plt.subplots(figsize=(h / 10, h * 3))
ax.axis("off")
fig.legend(
    handles=handles,
    loc="center",
    ncols=1,
    frameon=False,
    labelspacing=3,
)
fig.savefig(paths.figures / "regimes-legend.pdf")

# %%
# ====================================================================================
# COMPLEMENTARITY
# ====================================================================================

results = simulation.comp
simdata = results.drop(columns=["idx"]).groupby(["n", "k", "beta"]).mean().reset_index()

# %% Plot | Clustering ---------------------------------------------------------------

h = PLOT_HEIGHT
fig, axes = plt.subplots(
    ncols=(ncols := len(params.k)), figsize=(h * ncols, h), sharex=True, sharey=True
)

for ax, gdf in zip(axes.flat, simdata.groupby("k"), strict=True):
    k, gdf = gdf
    gdf["clustering"] += 1e-16  # Avoid log(0) issues
    y = "clustering"
    (
        so.Plot(gdf, x="n", y=y, color="beta", marker="beta")
        .add(so.Line(), so.Dodge(by=y), legend=False)
        .add(so.Dot(), so.Dodge(by=y), legend=False)
        .scale(
            color=so.Nominal(values=COLORS),
            marker=so.Nominal(values=MARKERS),
        )
        .on(ax)
        .plot()
    )
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"k={k}")
    ax.set_ylim(top=1.0)

fig.supylabel(r"clustering", x=0.02)
fig.tight_layout()
fig.savefig(paths.figures / "regimes-comp-clust.pdf")

# %% Plot | Q-Clustering -------------------------------------------------------------

h = PLOT_HEIGHT
fig, axes = plt.subplots(
    ncols=(ncols := len(params.k)), figsize=(h * ncols, h), sharex=True, sharey=True
)

for ax, gdf in zip(axes.flat, simdata.groupby("k"), strict=True):
    k, gdf = gdf
    y = "qclustering"
    (
        so.Plot(gdf, x="n", y=y, color="beta", marker="beta")
        .add(so.Line(), so.Dodge(by=y), legend=False)
        .add(so.Dot(), so.Dodge(by=y), legend=False)
        .scale(
            color=so.Nominal(values=COLORS),
            marker=so.Nominal(values=MARKERS),
        )
        .on(ax)
        .plot()
    )
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"k={k}")
    ax.set_ylim(top=1.0)

fig.supylabel(r"$q$-clustering", x=0.02)
fig.tight_layout()
fig.savefig(paths.figures / "regimes-comp-qclust.pdf")

# %% Plot | Average Path Length ------------------------------------------------------

h = PLOT_HEIGHT
fig, axes = plt.subplots(
    ncols=(ncols := len(params.k)), figsize=(h * ncols, h), sharex=True, sharey=True
)

for ax, gdf in zip(axes.flat, simdata.groupby("k"), strict=True):
    k, gdf = gdf
    y = "average_path_length"
    (
        so.Plot(gdf, x="n", y=y, color="beta", marker="beta")
        .add(so.Line(), so.Dodge(by=y), legend=False)
        .add(so.Dot(), so.Dodge(by=y), legend=False)
        .scale(
            color=so.Nominal(values=COLORS),
            marker=so.Nominal(values=MARKERS),
        )
        .on(ax)
        .plot()
    )
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"k={k}")

fig.supylabel(r"Avg. path length (L)", x=0.02)
fig.tight_layout()
fig.savefig(paths.figures / "regimes-comp-paths.pdf")

# %% ---------------------------------------------------------------------------------
