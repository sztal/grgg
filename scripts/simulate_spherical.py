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


def sample_stats(args: tuple[GRGG, int, int]) -> list[dict[str, float | int]]:
    model, seed, idx = args
    local_seed = np.random.SeedSequence([seed, idx])
    G = model.sample(random_state=local_seed).G
    coefs = PathCensus(G, parellel=False).coefs("global")
    density = G.density()
    output = {
        "n": model.n_nodes,
        "k": model.manifold.dim,
        "kbar": None,
        "kbar_est": model.kbar,
        "kbar_net": (model.n_nodes - 1) * density,
        "beta": model.kernels[0].beta,
        "logspace": model.kernels[0].logspace,
        "density": density,
        "clustering": coefs["sim_g"],
        "qclustering": coefs["comp_g"],
        "average_path_length": G.average_path_length(),
    }
    return output


ParamsT = tuple[type[AbstractGeometricKernel], float, int, int, float, bool]


def prepare_model(args: ParamsT) -> GRGG:
    kernel, kbar, n, k, beta, logspace = args
    model = GRGG(n, k).add_kernel(kbar, kernel, beta=beta, logspace=logspace)
    return model


def simulate(
    kernel: type[AbstractGeometricKernel],
    params: DictConfig,
    seed: int = config.seed,
    n_jobs: int = min(config.n_jobs, max(cpu_count() - 2, 1)),
) -> pd.DataFrame:
    pars = product(
        [kernel], [params.kbar], params.n, params.k, params.beta, params.logspace
    )
    models = pqdm(list(pars), prepare_model, n_jobs=n_jobs)
    args = product(models, [seed], range(params.n_rep))
    args = [(*a[:2], i) for i, a in enumerate(args)]
    results = pqdm(args, sample_stats, n_jobs=n_jobs)
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
