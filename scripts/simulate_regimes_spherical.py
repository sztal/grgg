# %% ---------------------------------------------------------------------------------

import os
from itertools import product
from multiprocessing import cpu_count
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from pathcensus import PathCensus
from pqdm.processes import pqdm

from grgg import GRGG, make_paths
from grgg.kernels import AbstractGeometricKernel, Complementarity, Similarity

paths = make_paths()
params = SimpleNamespace(
    n=10 ** np.arange(2, 6),  # Number of nodes
    k=2 ** np.arange(1, 5),  # Surface dimensions of the sphere
    beta=[0.0, 0.5, 1.5, 2.5, 5.0, 10.0, np.inf],  # kernel 'beta' values
    logspace=[True, False],
    kbar=10,  # Average degree
    nrep=10,  # Number of replications
)
n_jobs = min(os.environ.get("GRGG_N_JOBS", 10), max(cpu_count() - 2, 1))
seed = 421765311

# %% Simulation function -------------------------------------------------------------

KernelT = type[AbstractGeometricKernel]
ParamsT = tuple[KernelT, float, int, int, float, bool, int, int, int]


def simulate_one(args: ParamsT) -> list[dict[str, float | int]]:
    kernel, kbar, n, k, beta, logspace, nrep, seed, idx = args
    model = GRGG(n, k).add_kernel(kbar, kernel, beta=beta, logspace=logspace)
    local_seed = np.random.SeedSequence([seed, idx])
    rng = np.random.default_rng(local_seed)
    # Run simulations
    results = []
    for _ in range(nrep):
        G = model.sample(random_state=rng).G
        coefs = PathCensus(G).coefs("global").iloc[0]
        density = G.density()
        output = {
            "n": model.n_nodes,
            "k": model.manifold.dim,
            "kbar": kbar,
            "kbar_est": model.kbar,
            "kbar_net": (model.n_nodes - 1) * density,
            "beta": model.kernels[0].beta,
            "logspace": model.kernels[0].logspace,
            "density": density,
            "clustering": coefs["sim_g"],
            "qclustering": coefs["comp_g"],
            "average_path_length": G.average_path_length(),
        }
        results.append(output)
    return results


def simulate(
    kernel: type[AbstractGeometricKernel],
    params: SimpleNamespace,
    seed: int = seed,
    n_jobs: int = n_jobs,
) -> pd.DataFrame:
    kbar = params.kbar
    nrep = params.nrep
    pars = product(
        [kernel],
        [kbar],
        params.n,
        params.k,
        params.beta,
        params.logspace,
        [nrep],
        [seed],
    )
    pars = [(*p, i) for i, p in enumerate(pars)]
    results = []
    for result in pqdm(pars, simulate_one, n_jobs=n_jobs):
        results.extend(result)
    results = pd.DataFrame(results)
    return results


# %% Similarity ----------------------------------------------------------------------

sim = simulate(Similarity, params)

# %% Complementarity -----------------------------------------------------------------

comp = simulate(Complementarity, params)

# %% Save results --------------------------------------------------------------------

results = SimpleNamespace(params=params, sim=sim, comp=comp)
joblib.dump(results, paths.proc / "regimes-spherical.pkl", compress=9)

# %% ---------------------------------------------------------------------------------
