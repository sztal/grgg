import numpy as np
import pytest

from grgg import GRGG


class TestSimilarity:
    def test_kbar(self, model_sim: GRGG, kbar: float) -> None:
        model = model_sim
        model.calibrate(kbar)
        assert model.kbar == pytest.approx(kbar, rel=1e-2)
        K = [model.sample().A.sum(axis=1).mean() for _ in range(10)]
        assert np.mean(K) == pytest.approx(kbar, rel=1e-1)


class TestComplementarity:
    def test_kbar(self, model_comp: GRGG, kbar: float) -> None:
        model = model_comp
        model.calibrate(kbar)
        assert model.kbar == pytest.approx(kbar, rel=1e-2)
        K = [model.sample().A.sum(axis=1).mean() for _ in range(10)]
        assert np.mean(K) == pytest.approx(kbar, rel=1e-1)


class TestSimilarityComplementarity:
    def test_kbar(self, model_simcomp: GRGG, kbar: float, weights: np.ndarray) -> None:
        model = model_simcomp
        model.calibrate(kbar, weights)
        assert model.kbar == pytest.approx(kbar, rel=1e-2)
        K = [model.sample().A.sum(axis=1).mean() for _ in range(10)]
        assert np.mean(K) == pytest.approx(kbar, rel=1e-1)
        q = weights[0] if weights is not None else 0.5
        kbar = sum(m.kbar for m in model.submodels)
        assert model[0].kbar == pytest.approx(kbar * q, rel=1e-1)
        assert model[1].kbar == pytest.approx(kbar * (1 - q), rel=1e-1)
