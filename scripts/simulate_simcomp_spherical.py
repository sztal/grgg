# %% ---------------------------------------------------------------------------------

from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from pathcensus import PathCensus
from tqdm.auto import tqdm, trange

from grgg import GRGG, Complementarity, Similarity, make_paths

paths = make_paths()
gridsize = 30
params = SimpleNamespace(
    n=1000,  # Number of nodes
    k=2 ** np.arange(4),  # Surface dimensions of the sphere
    beta_s=np.linspace(0, 10, gridsize),  # 'beta' values for the similarity kernel
    beta_c=np.linspace(0, 10, gridsize),  # 'beta' values for the complementarity kernel
    q=np.linspace(0, 1, gridsize),  # Relative range of similarity
    logspace=[True, False],  # Whether to use log distance
    kbar=10,  # Average degree
    nrep=10,  # Number of replications
)
rng = np.random.default_rng(45235791)

# %% Simulation function -------------------------------------------------------------


def simulate(
    params: SimpleNamespace,
    rng: np.random.Generator,
) -> pd.DataFrame:
    results = []
    for beta_s in tqdm(params.beta_s):
        for beta_c in tqdm(params.beta_c, leave=False):
            for q in tqdm(params.q, leave=False):
                for k in tqdm(params.k, leave=False):
                    for logspace in tqdm(params.logspace, leave=False):
                        kbar = params.kbar
                        rgg = (
                            GRGG.from_n(n=params.n, k=k)
                            .set_kernel(
                                Similarity, kbar=kbar, beta=beta_s, logspace=logspace
                            )
                            .set_kernel(
                                Complementarity,
                                kbar=kbar,
                                beta=beta_c,
                                logspace=logspace,
                            )
                            .calibrate(kbar=params.kbar, q=q)
                        )
                        for i in trange(params.nrep, leave=False):
                            G = rgg.sample(seed=rng).G
                            coefs = PathCensus(G).coefs("global").iloc[0]
                            output = {
                                "n": params.n,
                                "k": k,
                                "beta_s": beta_s,
                                "beta_c": beta_c,
                                "q": q,
                                "logspace": logspace,
                                "idx": i,
                                "density": G.density(),
                                "clustering": coefs["sim_g"],
                                "qclustering": coefs["comp_g"],
                                "average_path_length": G.average_path_length(),
                            }
                            results.append(output)
    results = pd.DataFrame(results)
    return results


# %% Simulate ------------------------------------------------------------------------

simcomp = simulate(params, rng)

# %% Save results --------------------------------------------------------------------

results = SimpleNamespace(params=params, simcomp=simcomp)
joblib.dump(results, paths.proc / "simcomp-spherical.pkl", compress=9)

# %% ---------------------------------------------------------------------------------
