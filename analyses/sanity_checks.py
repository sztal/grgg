# %% ---------------------------------------------------------------------------------

import igraph as ig
import numpy as np
from pathcensus import PathCensus

from grgg import GRGG, Complementarity, Similarity, options  # noqa

N = 1000  # number of nodes
KBAR = 10.0  # average degree
igraph_plot_kwargs = {
    "vertex_size": 5,
    "edge_width": 0.1,
}

# %% Sanity check | Similarity and transitivity --------------------------------------

rgg = GRGG.from_n(n=N, k=2).set_kernel(Similarity, kbar=KBAR)

assert np.isclose(
    rgg.kbar, KBAR, atol=1e-2
), "Average degree does not match the expected value."

G = rgg.sample().G
sim = PathCensus(G).coefs("global").iloc[0]

assert np.isclose(
    sim["sim_g"], G.transitivity_undirected(), atol=1e-2
), "Similarity coefficient does not match the transitivity of the graph."

sim  # noqa # type: ignore

# %% ----------------------------------------------------------------------------------

ig.plot(G, **igraph_plot_kwargs)

# %% Sanity check | Complementarity and transitivity ---------------------------------

rgg = GRGG.from_n(n=N, k=2).set_kernel(Complementarity, kbar=KBAR)

assert np.isclose(
    rgg.kbar, KBAR, atol=1e-2
), "Average degree does not match the expected value."

G = rgg.sample().G
comp = PathCensus(G).coefs("global").iloc[0]

assert G.transitivity_undirected() <= 0.02, "Transitivity of the graph is too high"

assert comp["comp_g"] >= 0.2, "Complementarity coefficient is too low"

comp  # noqa # type: ignore

# %% ----------------------------------------------------------------------------------

ig.plot(G, **igraph_plot_kwargs)

# %% Sanity check | Similarity-Complementarity RGG model ------------------------------

q = 0.1  # relative weight of the similarity kernel
rgg = (
    GRGG.from_n(n=N, k=2)
    .set_kernel(Similarity, kbar=KBAR * q)
    .set_kernel(Complementarity, kbar=KBAR * (1 - q))
    .calibrate(KBAR, q=q)
)

assert np.isclose(
    rgg.kbar, KBAR, atol=1e-2
), "Average degree does not match the expected value."

G = rgg.sample().G
simcomp = PathCensus(G).coefs("global").iloc[0]

assert simcomp["sim_g"] >= 0.1, "Similarity coefficient is too low"
assert simcomp["comp_g"] >= 0.1, "Complementarity coefficient is too low"

simcomp  # noqa # type: ignore

# %% ----------------------------------------------------------------------------------

ig.plot(G, **igraph_plot_kwargs)

# %% ---------------------------------------------------------------------------------
