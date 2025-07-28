# %% ---------------------------------------------------------------------------------

from pathlib import Path
from types import SimpleNamespace

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathcensus import PathCensus
from tqdm.auto import tqdm, trange

from grgg import GRGG, Similarity, options

options.logdist = True  # logarithmic distance to allow for small-world effects

# Paths
paths = SimpleNamespace(here=Path(__file__).parent.absolute())
paths.root = paths.here.parent
paths.figures = paths.root / "figures"
paths.figures.mkdir(exist_ok=True, parents=True)

# Matplotlib settings
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["savefig.bbox"] = "tight"

N = [
    100,
    1000,
]  # Numbers of nodes
D = [1, 2, 4, 8]  # Surface dimensions of the sphere
K = 10  # Average degree
R = 10  # Number of replications
B = [  # 'beta' values for the kernels
    0.0,  # ER model
    0.5,
    1.1,
    1.5,
    3.0,
    np.inf,  # Hard RGG
]
rng = np.random.default_rng(425365311)

# ====================================================================================
# SIMILARITY
# ====================================================================================

# %% Simulate ------------------------------------------------------------------------

results = []
for b in tqdm(B):
    for d in tqdm(D, leave=False):
        for n in tqdm(N, leave=False):
            rgg = GRGG.from_n(n=n, k=d).set_kernel(Similarity, kbar=K, beta=b)
            for i in trange(R, leave=False):
                G = rgg.sample(sparse=False, seed=rng).G
                coefs = PathCensus(G).coefs("global").iloc[0]
                results.append(
                    {
                        "n": n,
                        "k": d,
                        "beta": b,
                        "idx": i,
                        "density": G.density(),
                        "clustering": coefs["sim_g"],
                        "qclustering": coefs["comp_g"],
                        "average_path_length": G.average_path_length(),
                    }
                )

results = pd.DataFrame(results)
simdata = (
    results.groupby(["n", "k", "beta"])[
        ["density", "clustering", "qclustering", "average_path_length"]
    ]
    .mean()
    .reset_index()
)

# %% Plot | Clustering ---------------------------------------------------------------

fig, axes = plt.subplots(
    ncols=(ncols := len(D)), figsize=(10, 2), sharex=True, sharey=True
)

for ax, gdf in zip(axes.flat, simdata.groupby("k"), strict=True):
    k, gdf = gdf
    for bdf in gdf.groupby("beta"):
        beta, bdf = bdf
        label = f"β={beta:.2f}"
        ax.plot(bdf["n"], bdf["clustering"], "o-", label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0, 1)
    ax.set_title(f"k={k}")
axes.flatten()[-1].legend()

fig.tight_layout()

# %% Plot | Q-Clustering -------------------------------------------------------------

fig, axes = plt.subplots(
    ncols=(ncols := len(D)), figsize=(10, 2), sharex=True, sharey=True
)

for ax, gdf in zip(axes.flat, simdata.groupby("k"), strict=True):
    k, gdf = gdf
    for bdf in gdf.groupby("beta"):
        beta, bdf = bdf
        label = f"β={beta:.2f}"
        ax.plot(bdf["n"], bdf["qclustering"], "o-", label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0, 1)
    ax.set_title(f"k={k}")
axes.flatten()[-1].legend()

fig.tight_layout()

# %% Plot | Average Path Length ------------------------------------------------------

fig, axes = plt.subplots(
    ncols=(ncols := len(D)), figsize=(10, 2), sharex=True, sharey=True
)

for ax, gdf in zip(axes.flat, simdata.groupby("k"), strict=True):
    k, gdf = gdf
    for bdf in gdf.groupby("beta"):
        beta, bdf = bdf
        label = f"β={beta:.2f}"
        ax.plot(bdf["n"], bdf["average_path_length"], "o-", label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"k={k}")
axes.flatten()[-1].legend()

fig.tight_layout()


# %% ---------------------------------------------------------------------------------
