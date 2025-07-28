# %% ---------------------------------------------------------------------------------

from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from pathcensus import PathCensus
from tqdm.auto import tqdm, trange

from grgg import GRGG, make_paths
from grgg.kernels import AbstractGeometricKernel, Complementarity, Similarity

paths = make_paths()
params = SimpleNamespace(
    n=10 ** np.arange(2, 5),  # Number of nodes
    k=2 ** np.arange(3),  # Surface dimensions of the sphere
    beta=[0.0, 1.1, 1.5, 5.0, np.inf],  # 'beta' values for the kernels
    kbar=10,  # Average degree
    nrep=5,  # Number of replications
)
rng = np.random.default_rng(421765311)

# %% Simulation function -------------------------------------------------------------


def simulate(
    kernel: type[AbstractGeometricKernel],
    params: SimpleNamespace,
    rng: np.random.Generator,
) -> pd.DataFrame:
    results = []
    for beta in tqdm(params.beta):
        for k in tqdm(params.k, leave=False):
            for n in tqdm(params.n, leave=False):
                rgg = GRGG.from_n(n=n, k=k).set_kernel(
                    kernel, kbar=params.kbar, beta=beta
                )
                for i in trange(params.nrep, leave=False):
                    G = rgg.sample(seed=rng).G
                    coefs = PathCensus(G).coefs("global").iloc[0]
                    output = {
                        "n": n,
                        "k": k,
                        "beta": beta,
                        "idx": i,
                        "density": G.density(),
                        "clustering": coefs["sim_g"],
                        "qclustering": coefs["comp_g"],
                        "average_path_length": G.average_path_length(),
                    }
                    results.append(output)
    results = pd.DataFrame(results)
    return results


# %% Similarity ----------------------------------------------------------------------

sim = simulate(Similarity, params, rng)

# %% Complementarity -----------------------------------------------------------------

comp = simulate(Complementarity, params, rng)

# %% Save results --------------------------------------------------------------------

results = SimpleNamespace(params=params, sim=sim, comp=comp)
joblib.dump(results, paths.proc / "regimes-spherical.pkl", compress=9)

# %% ---------------------------------------------------------------------------------
