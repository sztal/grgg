from math import isclose

import numpy as np
import pandas as pd
import pytest
from pathcensus import PathCensus

from grgg import GRGG, Complementarity, Similarity


class TestSimilarity:
    def test_kbar(self, model_sim: GRGG, kbar: float) -> None:
        model = model_sim
        model.calibrate(kbar)
        assert model.kbar == pytest.approx(kbar, rel=1e-2)
        K = [model.sample().A.sum(axis=1).mean() for _ in range(20)]
        assert np.mean(K) == pytest.approx(kbar, rel=1e-1)

    def test_theory(
        self,
        model: GRGG,
        kbar: float,
        kernel_sim: tuple[float, bool, type[Similarity]],
        rng: np.random.Generator,
    ) -> None:
        beta, logspace, kernel_type = kernel_sim
        model = model.add_kernel(kbar, kernel_type, beta=beta, logspace=logspace)
        density = kbar / (model.n_nodes - 1)
        G = [model.sample(random_state=rng).G for _ in range(20)]
        clust = pd.Series([g.transitivity_undirected() for g in G]).median()
        kernel = model.kernels[0]
        if kernel.beta == 0:
            isclose(clust, density, rel_tol=1e-1)
        elif kernel.beta == 3.0:
            assert clust > 0.1
        else:
            assert clust > 0.3


class TestComplementarity:
    def test_kbar(self, model_comp: GRGG, kbar: float) -> None:
        model = model_comp
        model.calibrate(kbar)
        assert model.kbar == pytest.approx(kbar, rel=1e-2)
        K = [model.sample().A.sum(axis=1).mean() for _ in range(20)]
        assert np.mean(K) == pytest.approx(kbar, rel=1e-1)

    def test_theory(
        self,
        model: GRGG,
        kbar: float,
        kernel_comp: tuple[float, bool, type[Complementarity]],
        rng: np.random.Generator,
    ) -> None:
        beta, logspace, kernel_type = kernel_comp
        model = model.add_kernel(kbar, kernel_type, beta=beta, logspace=logspace)
        density = kbar / (model.n_nodes - 1)
        G = [model.sample(random_state=rng).G for _ in range(20)]
        clust = pd.Series([PathCensus(g).complementarity("global") for g in G]).median()
        kernel = model.kernels[0]
        if kernel.beta == 0:
            isclose(clust, density, rel_tol=1e-1)
        elif kernel.beta == 3.0:
            assert clust > 0.05
        else:
            assert clust > 0.2


class TestSimilarityComplementarity:
    def test_kbar(self, model_simcomp: GRGG, kbar: float, weights: np.ndarray) -> None:
        model = model_simcomp
        model.calibrate(kbar, weights)
        assert model.kbar == pytest.approx(kbar, rel=1e-2)
        K = [model.sample().A.sum(axis=1).mean() for _ in range(50)]
        assert np.mean(K) == pytest.approx(kbar, rel=1e-1)
        q = weights[0] if weights is not None else 0.5
        kbar = sum(m.kbar for m in model.submodels)
        assert model[0].kbar == pytest.approx(kbar * q, rel=1e-1)
        assert model[1].kbar == pytest.approx(kbar * (1 - q), rel=1e-1)

    def test_theory(
        self,
        model: GRGG,
        kbar: float,
        kernel_sim: tuple[float, bool, type[Similarity]],
        kernel_comp: tuple[float, bool, type[Complementarity]],
        rng: np.random.Generator,
    ) -> None:
        beta_s, logspace_s, kernel_type_s = kernel_sim
        beta_c, logspace_c, kernel_type_c = kernel_comp
        model = (
            model.add_kernel(kbar, kernel_type_s, beta=beta_s, logspace=logspace_s)
            .add_kernel(kbar, kernel_type_c, beta=beta_c, logspace=logspace_c)
            .calibrate(kbar, weights=[1, 2])
        )
        density = model.density
        clust_sim = []
        clust_comp = []
        for _ in range(20):
            g = model.sample(random_state=rng).G
            clust_sim.append(g.transitivity_undirected())
            clust_comp.append(PathCensus(g).complementarity("global"))
        clust_sim = pd.Series(clust_sim).median()
        clust_comp = pd.Series(clust_comp).median()
        ksim = model.kernels[0]
        kcomp = model.kernels[1]
        if ksim.beta == 0:
            assert isclose(clust_sim, density, rel_tol=1e-1, abs_tol=1e-1)
        else:
            assert clust_sim > 0.02
        if kcomp.beta == 0:
            assert (
                isclose(clust_comp, density, rel_tol=1e-1, abs_tol=0.03)
                or clust_comp < density
            )
        else:
            assert clust_comp > 0.01
