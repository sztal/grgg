import numpy as np
import pytest

from grgg.quantize import ArrayQuantizer


class TestArrayQuantizer:
    seeds = np.random.SeedSequence(17).spawn(25)
    std_per_bin = 1e-6
    xmax = [5, 20]

    def allclose(
        self, A: np.ndarray, B: np.ndarray, *, rtol: float = 1e-3, atol: float = 1e-2
    ) -> bool:
        return np.allclose(A, B, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("xmax", xmax)
    @pytest.mark.parametrize("seed", seeds)
    def test_exact_lossy_dequantization(
        self, xmax: int, seed: np.random.SeedSequence
    ) -> None:
        rng = np.random.default_rng(seed)
        Y = rng.integers(0, xmax, size=(100, 2))
        Q = ArrayQuantizer(std_per_bin=self.std_per_bin, lossy=True)
        _ = Q.quantize(Y)
        Xd = Q.dequantize()
        assert self.allclose(Xd, Y)

    @pytest.mark.parametrize("xmax", xmax)
    @pytest.mark.parametrize("size", [1, 5])
    @pytest.mark.parametrize("lossy", [True, False])
    @pytest.mark.parametrize("seed", seeds)
    def test_indexed_dequantization(
        self, xmax: int, size: int, lossy: bool, seed: np.random.SeedSequence
    ) -> None:
        rng = np.random.default_rng(seed)
        Y = rng.integers(0, xmax, size=(100, 2))
        Q = ArrayQuantizer(std_per_bin=self.std_per_bin, lossy=lossy)
        Xq = Q.quantize(Y)
        idx = rng.choice(len(Xq), size=size, replace=False)
        idx.sort()
        invidx = Q.invmap_ids(idx)
        Xd = Q.dequantize(Xq[idx], idx)
        assert self.allclose(Xd, Y[invidx])

    @pytest.mark.parametrize("xmax", xmax)
    @pytest.mark.parametrize("seed", seeds)
    def test_invmap_ids_invariant(
        self, xmax: int, seed: np.random.SeedSequence
    ) -> None:
        rng = np.random.default_rng(seed)
        Y = rng.integers(0, xmax, size=(100, 2))
        quantizer = ArrayQuantizer(std_per_bin=self.std_per_bin, lossy=True)
        Xq = quantizer.quantize(Y)
        i = rng.choice(len(Xq), size=1, replace=False)[0]
        invidx = quantizer.invmap_ids(i)
        assert len(invidx) == quantizer.counts[i]
