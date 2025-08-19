import pytest

from grgg import GRGG


class TestSimilarityRGG:
    def test_kbar(self, model_sim: GRGG, kbar: float) -> None:
        model_sim.calibrate(kbar)
        assert model_sim.kbar == pytest.approx(kbar, rel=1e-2)
