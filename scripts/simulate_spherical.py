# %% ---------------------------------------------------------------------------------

from itertools import product
from multiprocessing import cpu_count
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from pathcensus import PathCensus
from pqdm.processes import pqdm

from grgg import GRGG
from grgg.kernels import AbstractGeometricKernel, Complementarity, Similarity
from grgg.project import config, paths

config = config.simulate.spherical

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
    params: DictConfig,
    seed: int = config.seed,
    n_jobs: int = min(config.n_jobs, max(cpu_count() - 2, 1)),
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

sim = simulate(Similarity, config.params)

# %% Complementarity -----------------------------------------------------------------

comp = simulate(Complementarity, config.params)

# %% Save results --------------------------------------------------------------------

results = SimpleNamespace(params=SimpleNamespace(**config.params), sim=sim, comp=comp)
joblib.dump(results, paths.proc / "spherical.pkl", compress=9)

# %% ---------------------------------------------------------------------------------
