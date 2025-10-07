# %% [markdown]

# Importance sampling estimators for undirected soft configuration models
# =======================================================================

# This notebook presents an analysis and confirmation of the effectivness
# of the proposed importance sampling estimators of motifs counts and motif-based
# statistics (e.g. triangle- and quadrangle-clustering coefficients) for undirected
# binary configuration models (UBCMs).
#
# We will consider two cases:
# 1. Networks with $n = 100$ nodes
# 2. Networks with $n = 1000$ nodes
#
# For each case, we will look at a $2 \times 2$ grid of subcases.
# One axis will be the density:
# - Low average degree ($\langle k \rangle \approx 5-10$)
# - High average degree ($\langle k \rangle \approx 50$)
#
# And the other axis will be degree heterogeneity:
# - Constant $\mu_i$'s (low heterogeneity / no heteogeneity in expectation)
# - Gaussian with nonzero variance $\mu_i$'s (high heterogeneity)
#
#
# We run tests for the following motifs:
# - Triangles
# - Quadrangles
# - Quadrangle wedges
# - Quadrangle heads
#
# Then, we will also look at the following motif-based statistics:
# - Triangle clustering coefficient
# - Triangle closure coefficient
# - Quadrangle clustering coefficient
# - Quadrangle closure coefficient
#
# For each combination of parameters, we will compute the exact expected value
# and then importance sampling estimates with varying number of samples
# repeated 5 times to account for variability.
#
# Tests for triangle wedge and head paths are not required, since these
# path motifs can be computed very efficiently without sampling.
#
#
# -------
# ## Idea
#
# In random graph models with conditionally independent
# edge probabilities and node-degree parameters edge probability $p_{ij}$
# (conditional on the parameters of the focal node $i$) is approximately propotional
# to the degree of node $j$. Now, when computing functions which are expressible
# as sums over neighbors, and then neighbors of neighbors and so on, it is possible to
# use importance sampling to sample only those neighbors which will have largest
# contributions to the overall sum, and then just appropriately reweight the obtained
# sum. This allows for reducing computational complexity almost arbitrarily from any
# $O(n^k)$ to just $O(n)$ for any motif of size $k$.
#
# Even more remarkably, the accuracy of this approximation depends on the extent to
# which $p_{ij}$'s are proportional to the degrees, and this proportionality is
# strongest for high-degree nodes. As a result, it is enough to sample only a limited
# number of most connected neighbors to obtain very good (nearly) unbiased estimates.
# As a matter of fact, at certain point, increasing the number of samples
# actually decreases the validity of the estimates, since it introduces
# more bias from the non-proportionality of $p_{ij}$'s to degrees
# (and the rescaling formula assumes perfect proportionality).


# %% ---------------------------------------------------------------------------------

import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from grgg import RandomGenerator, UndirectedRandomGraph
from grgg.project import paths, theme

theme(
    {
        "figure.figsize": (4, 4),
    }
)
rng = RandomGenerator(303)  # Random generator with a fixed seed for reproducibility

N_REPS = 5  # Number of repetitions for sampling-based estimators

FIGS = paths.figures / "random_graphs" / "importance-sampling"
FIGS.mkdir(parents=True, exist_ok=True)


def define_stat(func):
    def compute_stats(n_samples=-1, *args, n_reps=1, **kwargs):
        output = jnp.stack(
            [func(*args, n_samples=n_samples, **kwargs) for _ in range(n_reps)]
        )
        return output

    return compute_stats


# %% [markdown]
# ## Motif counts
#
# We start by looking at the estimators of expected motif counts,
# as these are building blocks from which other motif-based statistics
# can be derived.
#
# ### Networks with $n = 100$

# %% ---------------------------------------------------------------------------------

n = 100
n_samples_grid = [1, 10, 20, 50, 90]
motifs = ["triangle", "quadrangle", "qwedge", "qhead"]
# Note that we must defined 'n_samples_grid' as a standard Python list,
# as a JAX array would cause tracing issues in the loops below.

# %% [markdown]
# ### Low average degree, degree homogeneity
#
# The homogeneous case is trivial, since in this case all nodes
# are identical and importance sampling is not really needed.
# However, it serves as a good sanity check, as it should always reproduce
# the exact results.

# %% ---------------------------------------------------------------------------------

# Create a homogeneous model, but implemented as a heterogeneous one
# with all mu's being equal.
model = UndirectedRandomGraph(n, mu=jnp.ones(n) * -1.46)
model.nodes.degree().mean()

# %% ---------------------------------------------------------------------------------

stats = {motif: define_stat(n, getattr(model.nodes.motifs, motif)) for motif in motifs}
exact = {motif: stat() for motif, stat in stats.items()}
approx = {
    motif: jnp.stack([stat(n_samples=n_i, n_reps=5) for n_i in tqdm(n_samples_grid)])
    for motif, stat in stats.items()
}

# %% [markdown]
#
# As expected, in this case there is a perfect agreement between
# the expected values and their importance sampling estimates

# %% ---------------------------------------------------------------------------------

fig, axes = plt.subplots(
    nrows=len(motifs),
    ncols=len(n_samples_grid),
    figsize=(10, 8),
    sharex=True,
    sharey=True,
)
for motif, axrow in zip(motifs, axes, strict=True):
    E = exact[motif]
    axrow[0].set_ylabel(motif, fontsize="x-large")
    for n_samples, X, ax in zip(
        n_samples_grid, approx[motif], axrow.flatten(), strict=True
    ):
        E, X = jnp.broadcast_arrays(E, X)
        ax.scatter(E.flatten(), X.flatten(), alpha=0.2)
        if ax in axes[0]:
            ax.set_title(f"$n_s = {n_samples}$")
        ax.set_xscale("log")
        ax.set_yscale("log")


kbar = model.nodes.degree().mean()
fig.text(
    0.12,
    0.96,
    rf"$n = {n}, \langle{{k}}\rangle \approx {kbar:.1f}$,"
    "\n"
    "degree-"
    + ("homogeneous" if jnp.unique(model.mu.value).size == 1 else "heterogeneous"),
    fontsize="large",
    bbox={"facecolor": "white", "alpha": 0.8},
)
fig.suptitle("Number of importance samples", x=0.52, y=0.99, fontsize="xx-large")
fig.supxlabel("Expected value", fontsize="xx-large")
fig.supylabel("Importance sampling estimate", fontsize="xx-large")
fig.tight_layout()
fig.savefig(FIGS / "motifs-100-low-degree-homogeneous.png")

# %% [markdown]
# #### High average degree, degree homogeneity
#
# The same results hold for the high-degree homogeneous case.

# %% ---------------------------------------------------------------------------------

model = UndirectedRandomGraph(n, mu=jnp.ones(n) * 0)
model.nodes.degree().mean()

# %% ---------------------------------------------------------------------------------

stats = {motif: define_stat(n, getattr(model.nodes.motifs, motif)) for motif in motifs}
exact = {motif: stat() for motif, stat in stats.items()}
approx = {
    motif: jnp.stack([stat(n_samples=n_i, n_reps=5) for n_i in tqdm(n_samples_grid)])
    for motif, stat in stats.items()
}

# %% ---------------------------------------------------------------------------------

fig, axes = plt.subplots(
    nrows=len(motifs),
    ncols=len(n_samples_grid),
    figsize=(10, 8),
    sharex=True,
    sharey=True,
)
for motif, axrow in zip(motifs, axes, strict=True):
    E = exact[motif]
    axrow[0].set_ylabel(motif, fontsize="x-large")
    for n_samples, X, ax in zip(
        n_samples_grid, approx[motif], axrow.flatten(), strict=True
    ):
        E, X = jnp.broadcast_arrays(E, X)
        ax.scatter(E.flatten(), X.flatten(), alpha=0.2)
        if ax in axes[0]:
            ax.set_title(f"$n_s = {n_samples}$")
        ax.set_xscale("log")
        ax.set_yscale("log")


kbar = model.nodes.degree().mean()
fig.text(
    0.12,
    0.96,
    rf"$n = {n}, \langle{{k}}\rangle \approx {kbar:.1f}$,"
    "\n"
    "degree-"
    + ("homogeneous" if jnp.unique(model.mu.value).size == 1 else "heterogeneous"),
    fontsize="large",
    bbox={"facecolor": "white", "alpha": 0.8},
)
fig.suptitle("Number of importance samples", x=0.52, y=0.99, fontsize="xx-large")
fig.supxlabel("Expected value", fontsize="xx-large")
fig.supylabel("Importance sampling estimate", fontsize="xx-large")
fig.tight_layout()
fig.savefig(FIGS / "motifs-100-high-degree-homogeneous.pdf")


# %% [markdown]
# #### Low average degree, degree heterogeneity
#
# This is where things get interesting. Importance sampling works very well in this
# setting too, but best for moderate number of samples. For very low number of samples
# the estimates can be quite noisy, but unbiased. For very high number of samples,
# the noise is reduced, but bias creeps in, as the assumption of proportionality
# between edge probabilities and degrees is no longer valid.

# %% ---------------------------------------------------------------------------------

model = UndirectedRandomGraph(n, mu=rng.normal(n) * 2 - 2.5)
model.nodes.degree().mean()

# %% ---------------------------------------------------------------------------------

stats = {motif: define_stat(n, getattr(model.nodes.motifs, motif)) for motif in motifs}
exact = {motif: stat() for motif, stat in stats.items()}
approx = {
    motif: jnp.stack([stat(n_samples=n_i, n_reps=5) for n_i in tqdm(n_samples_grid)])
    for motif, stat in stats.items()
}

# %% [markdown]
#
# Below we are plots of the expected values vs importance sampling estimates
# for different number of samples. We also report the $R^2$ and relative
# Frobenius norm errors.

# %% ---------------------------------------------------------------------------------

fig, axes = plt.subplots(
    nrows=len(motifs),
    ncols=len(n_samples_grid),
    figsize=(10, 8),
    sharex=False,
    sharey=False,
)
for motif, axrow in zip(motifs, axes, strict=True):
    E = exact[motif]
    axrow[0].set_ylabel(motif, fontsize="x-large")
    for n_samples, X, ax in zip(
        n_samples_grid, approx[motif], axrow.flatten(), strict=True
    ):
        e, x = (x.flatten() for x in jnp.broadcast_arrays(E, X))
        # e = E.flatten()
        # x = X.mean(0)
        ax.scatter(e, x, alpha=0.2)
        es = jnp.unique(e)
        ax.plot(es, es, color="C1", ls="-", lw=2, zorder=99)
        if ax in axes[0]:
            ax.set_title(f"$n_s = {n_samples}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        # Some statistics
        rsq = jnp.corrcoef(e, x)[0, 1] ** 2
        err = jnp.linalg.norm(e - x) / jnp.linalg.norm(e)
        ax.annotate(
            rf"$R^2 = {rsq:.2f}$" + "\n" + rf"$\varepsilon\;\;\, = {err:.2f}$",
            (0.05, 0.95),
            xycoords="axes fraction",
            va="top",
            ha="left",
            fontsize="small",
        )

kbar = model.nodes.degree().mean()
fig.text(
    0.12,
    0.955,
    rf"$n = {n}, \langle{{k}}\rangle \approx {kbar:.1f}$,"
    "\n"
    "degree-"
    + ("homogeneous" if jnp.unique(model.mu.value).size == 1 else "heterogeneous"),
    fontsize="large",
    bbox={"facecolor": "white", "alpha": 0.8},
)
fig.suptitle("Number of importance samples", x=0.52, y=0.99, fontsize="xx-large")
fig.supxlabel("Expected value", fontsize="xx-large")
fig.supylabel("Importance sampling estimate", fontsize="xx-large")
fig.tight_layout()
fig.savefig(FIGS / "motifs-100-low-degree-heterogeneous.pdf")


# %% [markdown]
# #### High average degree, degree heterogeneity
#
# The same as before, but now with high average degree.
# The results are generally identical. However, the bias increases
# somewhat slower with the number of samples, since nodes have higher
# degrees in general, and thus the the proportionality assumption
# is approximately valid for the large subset of nodes.
#
# A pattern that emerges from this analysis, and which confirms the theory,
# behind this method, is that (up to fluctuations) $R^2$ increases with the
# number of samples, as the estimates become less noisy and the general trend
# remains linear, but the relative error first decreases, but then increases
# again, as bias becomes non-negligible and produces location shifts in the estimates.

# %% ---------------------------------------------------------------------------------

model = UndirectedRandomGraph(n, mu=rng.normal(n) * 2 + 0)
model.nodes.degree().mean()

# %% ---------------------------------------------------------------------------------

stat = define_stat(n, model.nodes.motifs.triangle)
stats = {motif: define_stat(n, getattr(model.nodes.motifs, motif)) for motif in motifs}
exact = {motif: stat() for motif, stat in stats.items()}
approx = {
    motif: jnp.stack([stat(n_samples=n_i, n_reps=5) for n_i in tqdm(n_samples_grid)])
    for motif, stat in stats.items()
}

# %% ---------------------------------------------------------------------------------

fig, axes = plt.subplots(
    nrows=len(motifs),
    ncols=len(n_samples_grid),
    figsize=(10, 8),
    sharex=False,
    sharey=False,
)
for motif, axrow in zip(motifs, axes, strict=True):
    E = exact[motif]
    axrow[0].set_ylabel(motif, fontsize="x-large")
    for n_samples, X, ax in zip(
        n_samples_grid, approx[motif], axrow.flatten(), strict=True
    ):
        e, x = (x.flatten() for x in jnp.broadcast_arrays(E, X))
        ax.scatter(e, x, alpha=0.2)
        es = jnp.unique(e)
        ax.plot(es, es, color="C1", ls="-", lw=2, zorder=99)
        if ax in axes[0]:
            ax.set_title(f"$n_s = {n_samples}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        # Some statistics
        rsq = jnp.corrcoef(e, x)[0, 1] ** 2
        err = jnp.linalg.norm(e - x) / jnp.linalg.norm(e)
        ax.annotate(
            rf"$R^2 = {rsq:.2f}$" + "\n" + rf"$\varepsilon\;\;\, = {err:.2f}$",
            (0.05, 0.95),
            xycoords="axes fraction",
            va="top",
            ha="left",
            fontsize="small",
        )

kbar = model.nodes.degree().mean()
fig.text(
    0.12,
    0.955,
    rf"$n = {n}, \langle{{k}}\rangle \approx {kbar:.1f}$,"
    "\n"
    "degree-"
    + ("homogeneous" if jnp.unique(model.mu.value).size == 1 else "heterogeneous"),
    fontsize="large",
    bbox={"facecolor": "white", "alpha": 0.8},
)
fig.suptitle("Number of importance samples", x=0.52, y=0.99, fontsize="xx-large")
fig.supxlabel("Expected value", fontsize="xx-large")
fig.supylabel("Importance sampling estimate", fontsize="xx-large")
fig.tight_layout()
fig.savefig(FIGS / "motifs-100-high-degree-heterogeneous.pdf")


# %% [markdown]
# ### Networks with $n = 1000$
#
# In this case we look only at the non-trivial heterogeneous cases.

# %% ---------------------------------------------------------------------------------

n = 1000
n_samples_grid = [1, 10, 20, 50, 100]
motifs = ["triangle", "quadrangle", "qwedge", "qhead"]
# Note that we must defined 'n_samples_grid' as a standard Python list,
# as a JAX array would cause tracing issues in the loops below.

# %% [markdown]
# #### Low average degree, degree heterogeneity
#
# The results are qualitatively the same as for $n=100$.
# The samples threshold for optimal results is slightly higher,
# but the overall picture remains unchanged.

# %% ---------------------------------------------------------------------------------

model = UndirectedRandomGraph(n, mu=rng.normal(n) * 2 - 4)
model.nodes.degree().mean()

# %% ---------------------------------------------------------------------------------

stat = define_stat(n, model.nodes.motifs.triangle)
stats = {motif: define_stat(n, getattr(model.nodes.motifs, motif)) for motif in motifs}
exact = {motif: stat() for motif, stat in stats.items()}
approx = {
    motif: jnp.stack([stat(n_samples=n_i, n_reps=5) for n_i in tqdm(n_samples_grid)])
    for motif, stat in stats.items()
}

# %% ---------------------------------------------------------------------------------

fig, axes = plt.subplots(
    nrows=len(motifs),
    ncols=len(n_samples_grid),
    figsize=(10, 8),
    sharex=False,
    sharey=False,
)
for motif, axrow in zip(motifs, axes, strict=True):
    E = exact[motif]
    axrow[0].set_ylabel(motif, fontsize="x-large")
    for n_samples, X, ax in zip(
        n_samples_grid, approx[motif], axrow.flatten(), strict=True
    ):
        e, x = (x.flatten() for x in jnp.broadcast_arrays(E, X))
        ax.scatter(e, x, alpha=0.2)
        es = jnp.unique(e)
        ax.plot(es, es, color="C1", ls="-", lw=2, zorder=99)
        if ax in axes[0]:
            ax.set_title(f"$n_s = {n_samples}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        # Some statistics
        rsq = jnp.corrcoef(e, x)[0, 1] ** 2
        err = jnp.linalg.norm(e - x) / jnp.linalg.norm(e)
        ax.annotate(
            rf"$R^2 = {rsq:.2f}$" + "\n" + rf"$\varepsilon\;\;\, = {err:.2f}$",
            (0.05, 0.95),
            xycoords="axes fraction",
            va="top",
            ha="left",
            fontsize="small",
        )

kbar = model.nodes.degree().mean()
fig.text(
    0.12,
    0.955,
    rf"$n = {n}, \langle{{k}}\rangle \approx {kbar:.1f}$,"
    "\n"
    "degree-"
    + ("homogeneous" if jnp.unique(model.mu.value).size == 1 else "heterogeneous"),
    fontsize="large",
    bbox={"facecolor": "white", "alpha": 0.8},
)
fig.suptitle("Number of importance samples", x=0.52, y=0.99, fontsize="xx-large")
fig.supxlabel("Expected value", fontsize="xx-large")
fig.supylabel("Importance sampling estimate", fontsize="xx-large")
fig.tight_layout()
fig.savefig(FIGS / "motifs-1000-low-degree-heterogeneous.pdf")


# %% [markdown]
# #### High average degree, degree heterogeneity
#
# The same as before, but now with high average degree.
# The results are generally identical, but again the bias
# starts to creep in even later.

# %% ---------------------------------------------------------------------------------

model = UndirectedRandomGraph(n, mu=rng.normal(n) * 2 - 2.8)
model.nodes.degree().mean()

# %% ---------------------------------------------------------------------------------

stat = define_stat(n, model.nodes.motifs.triangle)
stats = {motif: define_stat(n, getattr(model.nodes.motifs, motif)) for motif in motifs}
exact = {motif: stat() for motif, stat in stats.items()}
approx = {
    motif: jnp.stack([stat(n_samples=n_i, n_reps=5) for n_i in tqdm(n_samples_grid)])
    for motif, stat in stats.items()
}

# %% ---------------------------------------------------------------------------------

fig, axes = plt.subplots(
    nrows=len(motifs),
    ncols=len(n_samples_grid),
    figsize=(10, 8),
    sharex=False,
    sharey=False,
)
for motif, axrow in zip(motifs, axes, strict=True):
    E = exact[motif]
    axrow[0].set_ylabel(motif, fontsize="x-large")
    for n_samples, X, ax in zip(
        n_samples_grid, approx[motif], axrow.flatten(), strict=True
    ):
        e, x = (x.flatten() for x in jnp.broadcast_arrays(E, X))
        ax.scatter(e, x, alpha=0.2)
        es = jnp.unique(e)
        ax.plot(es, es, color="C1", ls="-", lw=2, zorder=99)
        if ax in axes[0]:
            ax.set_title(f"$n_s = {n_samples}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        # Some statistics
        rsq = jnp.corrcoef(e, x)[0, 1] ** 2
        err = jnp.linalg.norm(e - x) / jnp.linalg.norm(e)
        ax.annotate(
            rf"$R^2 = {rsq:.2f}$" + "\n" + rf"$\varepsilon\;\;\, = {err:.2f}$",
            (0.05, 0.95),
            xycoords="axes fraction",
            va="top",
            ha="left",
            fontsize="small",
        )

kbar = model.nodes.degree().mean()
fig.text(
    0.12,
    0.955,
    rf"$n = {n}, \langle{{k}}\rangle \approx {kbar:.1f}$,"
    "\n"
    "degree-"
    + ("homogeneous" if jnp.unique(model.mu.value).size == 1 else "heterogeneous"),
    fontsize="large",
    bbox={"facecolor": "white", "alpha": 0.8},
)
fig.suptitle("Number of importance samples", x=0.52, y=0.99, fontsize="xx-large")
fig.supxlabel("Expected value", fontsize="xx-large")
fig.supylabel("Importance sampling estimate", fontsize="xx-large")
fig.tight_layout()
fig.savefig(FIGS / "motifs-1000-high-degree-heterogeneous.pdf")


# %% [markdown]
# ## Motif-based statistics
#
# Finally, we look at the performance of importance sampling
# for motif-based statistics, such as clustering coefficients.
# Again, we look only at the non-trivial heterogeneous cases
# with $n = 100, 1000$ and low or high average degree.
#
# ### Networks with $n = 100$

# %% ---------------------------------------------------------------------------------

n = 100
n_samples_grid = [1, 10, 20, 50, 90]
statgroups = {
    "tstats": ["tclust", "tclosure", "similarity"],
    "qstats": ["qclust", "qclosure", "complementarity"],
}

# %% [markdown]
# #### Low average degree, degree heterogeneity

# %% ---------------------------------------------------------------------------------

model = UndirectedRandomGraph(n, mu=rng.normal(n) * 2 - 2.5)
model.nodes.degree().mean()

# %% ---------------------------------------------------------------------------------

statistics = {stat: define_stat(getattr(model.nodes, stat)) for stat in statgroups}
exact = {}
for group, names in statgroups.items():
    stats = statistics[group]()
    for i, name in enumerate(names):
        exact[name] = stats[..., i, :]
approx = {}
for group, names in statgroups.items():
    stats = jnp.stack(
        [
            statistics[group](n_samples=n_i, n_reps=N_REPS)
            for n_i in tqdm(n_samples_grid)
        ]
    )
    for i, name in enumerate(names):
        approx[name] = stats[..., i, :]

# %% [markdown]
#

# %% ---------------------------------------------------------------------------------

fig, axes = plt.subplots(
    nrows=len(exact),
    ncols=len(n_samples_grid),
    figsize=(10, 12),
    sharex=False,
    sharey=False,
)
for stat, axrow in zip(exact, axes, strict=True):
    E = exact[stat]
    axrow[0].set_ylabel(stat, fontsize="x-large")
    for n_samples, X, ax in zip(
        n_samples_grid, approx[stat], axrow.flatten(), strict=True
    ):
        e, x = (x.flatten() for x in jnp.broadcast_arrays(E, X))
        # x = X.mean(0)
        # e = E.flatten()
        ax.scatter(e, x, alpha=0.2)
        es = jnp.unique(e)
        ax.plot(es, es, color="C1", ls="-", lw=2, zorder=99)
        if ax in axes[0]:
            ax.set_title(f"$n_s = {n_samples}$")
        # Some statistics
        rsq = jnp.corrcoef(e, x)[0, 1] ** 2
        err = jnp.linalg.norm(e - x) / jnp.linalg.norm(e)
        ax.annotate(
            rf"$R^2 = {rsq:.2f}$" + "\n" + rf"$\varepsilon\;\;\, = {err:.2f}$",
            (0.05, 0.95),
            xycoords="axes fraction",
            va="top",
            ha="left",
            fontsize="small",
        )

kbar = model.nodes.degree().mean()
fig.text(
    0.12,
    0.97,
    rf"$n = {n}, \langle{{k}}\rangle \approx {kbar:.1f}$,"
    "\n"
    "degree-"
    + ("homogeneous" if jnp.unique(model.mu.value).size == 1 else "heterogeneous"),
    fontsize="large",
    bbox={"facecolor": "white", "alpha": 0.8},
)
fig.suptitle("Number of importance samples", x=0.52, y=0.99, fontsize="xx-large")
fig.supxlabel("Expected value", fontsize="xx-large")
fig.supylabel("Importance sampling estimate", fontsize="xx-large")
fig.tight_layout()
fig.savefig(FIGS / "stats-100-low-degree-heterogeneous.pdf")


# %% [markdown]
# #### High average degree, degree heterogeneity

# %% ---------------------------------------------------------------------------------

model = UndirectedRandomGraph(n, mu=rng.normal(n) * 2 - 0)
model.nodes.degree().mean()

# %% ---------------------------------------------------------------------------------

statistics = {stat: define_stat(getattr(model.nodes, stat)) for stat in statgroups}
exact = {}
for group, names in statgroups.items():
    stats = statistics[group]()
    for i, name in enumerate(names):
        exact[name] = stats[..., i, :]
approx = {}
for group, names in statgroups.items():
    stats = jnp.stack(
        [
            statistics[group](n_samples=n_i, n_reps=N_REPS)
            for n_i in tqdm(n_samples_grid)
        ]
    )
    for i, name in enumerate(names):
        approx[name] = stats[..., i, :]

# %% [markdown]
#
# The agreement between expected values and importance sampling estimates
# Is even better in this case.

# %% ---------------------------------------------------------------------------------

fig, axes = plt.subplots(
    nrows=len(exact),
    ncols=len(n_samples_grid),
    figsize=(10, 12),
    sharex=False,
    sharey=False,
)
for stat, axrow in zip(exact, axes, strict=True):
    E = exact[stat]
    axrow[0].set_ylabel(stat, fontsize="x-large")
    for n_samples, X, ax in zip(
        n_samples_grid, approx[stat], axrow.flatten(), strict=True
    ):
        e, x = (x.flatten() for x in jnp.broadcast_arrays(E, X))
        # x = X.mean(0)
        # e = E.flatten()
        ax.scatter(e, x, alpha=0.2)
        es = jnp.unique(e)
        ax.plot(es, es, color="C1", ls="-", lw=2, zorder=99)
        if ax in axes[0]:
            ax.set_title(f"$n_s = {n_samples}$")
        # Some statistics
        rsq = jnp.corrcoef(e, x)[0, 1] ** 2
        err = jnp.linalg.norm(e - x) / jnp.linalg.norm(e)
        ax.annotate(
            rf"$R^2 = {rsq:.2f}$" + "\n" + rf"$\varepsilon\;\;\, = {err:.2f}$",
            (0.05, 0.95),
            xycoords="axes fraction",
            va="top",
            ha="left",
            fontsize="small",
        )

kbar = model.nodes.degree().mean()
fig.text(
    0.12,
    0.97,
    rf"$n = {n}, \langle{{k}}\rangle \approx {kbar:.1f}$,"
    "\n"
    "degree-"
    + ("homogeneous" if jnp.unique(model.mu.value).size == 1 else "heterogeneous"),
    fontsize="large",
    bbox={"facecolor": "white", "alpha": 0.8},
)
fig.suptitle("Number of importance samples", x=0.52, y=0.99, fontsize="xx-large")
fig.supxlabel("Expected value", fontsize="xx-large")
fig.supylabel("Importance sampling estimate", fontsize="xx-large")
fig.tight_layout()
fig.savefig(FIGS / "stats-100-high-degree-heterogeneous.pdf")


# %% [markdown]
# ### Networks with $n = 1000$

# %% ---------------------------------------------------------------------------------

n = 1000
n_samples_grid = [1, 10, 20, 50, 100]
statistics = ["tstats", "qstats"]

# %% [markdown]
# #### Low average degree, degree heterogeneity

# %% ---------------------------------------------------------------------------------

model = UndirectedRandomGraph(n, mu=rng.normal(n) * 2 - 4.1)
model.nodes.degree().mean()

# %% ---------------------------------------------------------------------------------

statistics = {stat: define_stat(getattr(model.nodes, stat)) for stat in statgroups}
exact = {}
for group, names in statgroups.items():
    stats = statistics[group]()
    for i, name in enumerate(names):
        exact[name] = stats[..., i, :]
approx = {}
for group, names in statgroups.items():
    stats = jnp.stack(
        [
            statistics[group](n_samples=n_i, n_reps=N_REPS)
            for n_i in tqdm(n_samples_grid)
        ]
    )
    for i, name in enumerate(names):
        approx[name] = stats[..., i, :]

# %% [markdown]
#
# The agreement between expected values and importance sampling estimates
# Is even better in this case.

# %% ---------------------------------------------------------------------------------

fig, axes = plt.subplots(
    nrows=len(exact),
    ncols=len(n_samples_grid),
    figsize=(10, 12),
    sharex=False,
    sharey=False,
)
for stat, axrow in zip(exact, axes, strict=True):
    E = exact[stat]
    axrow[0].set_ylabel(stat, fontsize="x-large")
    for n_samples, X, ax in zip(
        n_samples_grid, approx[stat], axrow.flatten(), strict=True
    ):
        e, x = (x.flatten() for x in jnp.broadcast_arrays(E, X))
        # x = X.mean(0)
        # e = E.flatten()
        ax.scatter(e, x, alpha=0.2)
        es = jnp.unique(e)
        ax.plot(es, es, color="C1", ls="-", lw=2, zorder=99)
        if ax in axes[0]:
            ax.set_title(f"$n_s = {n_samples}$")
        # Some statistics
        rsq = jnp.corrcoef(e, x)[0, 1] ** 2
        err = jnp.linalg.norm(e - x) / jnp.linalg.norm(e)
        ax.annotate(
            rf"$R^2 = {rsq:.2f}$" + "\n" + rf"$\varepsilon\;\;\, = {err:.2f}$",
            (0.05, 0.95),
            xycoords="axes fraction",
            va="top",
            ha="left",
            fontsize="small",
        )

kbar = model.nodes.degree().mean()
fig.text(
    0.12,
    0.97,
    rf"$n = {n}, \langle{{k}}\rangle \approx {kbar:.1f}$,"
    "\n"
    "degree-"
    + ("homogeneous" if jnp.unique(model.mu.value).size == 1 else "heterogeneous"),
    fontsize="large",
    bbox={"facecolor": "white", "alpha": 0.8},
)
fig.suptitle("Number of importance samples", x=0.52, y=0.99, fontsize="xx-large")
fig.supxlabel("Expected value", fontsize="xx-large")
fig.supylabel("Importance sampling estimate", fontsize="xx-large")
fig.tight_layout()
fig.savefig(FIGS / "stats-1000-low-degree-heterogeneous.pdf")


# %% [markdown]
# #### High average degree, degree heterogeneity

# %% ---------------------------------------------------------------------------------

model = UndirectedRandomGraph(n, mu=rng.normal(n) * 2 - 2.7)
model.nodes.degree().mean()

# %% ---------------------------------------------------------------------------------

statistics = {stat: define_stat(getattr(model.nodes, stat)) for stat in statgroups}
exact = {}
for group, names in statgroups.items():
    stats = statistics[group]()
    for i, name in enumerate(names):
        exact[name] = stats[..., i, :]
approx = {}
for group, names in statgroups.items():
    stats = jnp.stack(
        [
            statistics[group](n_samples=n_i, n_reps=N_REPS)
            for n_i in tqdm(n_samples_grid)
        ]
    )
    for i, name in enumerate(names):
        approx[name] = stats[..., i, :]

# %% [markdown]
#
# The agreement between expected values and importance sampling estimates
# Is even better in this case.

# %% ---------------------------------------------------------------------------------

fig, axes = plt.subplots(
    nrows=len(exact),
    ncols=len(n_samples_grid),
    figsize=(10, 12),
    sharex=False,
    sharey=False,
)
for stat, axrow in zip(exact, axes, strict=True):
    E = exact[stat]
    axrow[0].set_ylabel(stat, fontsize="x-large")
    for n_samples, X, ax in zip(
        n_samples_grid, approx[stat], axrow.flatten(), strict=True
    ):
        e, x = (x.flatten() for x in jnp.broadcast_arrays(E, X))
        # x = X.mean(0)
        # e = E.flatten()
        ax.scatter(e, x, alpha=0.2)
        es = jnp.unique(e)
        ax.plot(es, es, color="C1", ls="-", lw=2, zorder=99)
        if ax in axes[0]:
            ax.set_title(f"$n_s = {n_samples}$")
        # Some statistics
        rsq = jnp.corrcoef(e, x)[0, 1] ** 2
        err = jnp.linalg.norm(e - x) / jnp.linalg.norm(e)
        ax.annotate(
            rf"$R^2 = {rsq:.2f}$" + "\n" + rf"$\varepsilon\;\;\, = {err:.2f}$",
            (0.05, 0.95),
            xycoords="axes fraction",
            va="top",
            ha="left",
            fontsize="small",
        )

kbar = model.nodes.degree().mean()
fig.text(
    0.12,
    0.97,
    rf"$n = {n}, \langle{{k}}\rangle \approx {kbar:.1f}$,"
    "\n"
    "degree-"
    + ("homogeneous" if jnp.unique(model.mu.value).size == 1 else "heterogeneous"),
    fontsize="large",
    bbox={"facecolor": "white", "alpha": 0.8},
)
fig.suptitle("Number of importance samples", x=0.52, y=0.99, fontsize="xx-large")
fig.supxlabel("Expected value", fontsize="xx-large")
fig.supylabel("Importance sampling estimate", fontsize="xx-large")
fig.tight_layout()
fig.savefig(FIGS / "stats-1000-high-degree-heterogeneous.pdf")
