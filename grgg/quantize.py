from typing import Any, Self

import numpy as np

from . import options
from .utils.discretizers import KMeansDiscretizer

__all__ = ("ArrayQuantizer",)


class ArrayQuantizer:
    """Quantizer for numerical arrays.

    Attributes
    ----------
    discretizer
        The discretizer used for quantization.
    lossy
        Whether the quantization is lossy.
        If `False`, then a copy of tge original array is stored,
        so it is returned exactly by `dequantize()`.
        Otherwise, the origina array is reconstructed from the bins,
        which may introduce some information loss.
    """

    def __init__(self, *, lossy: bool = False, **kwargs: Any) -> None:
        """Initialization method.

        `**kwargs` are passed to :class:`grgg.utils.discretizers.KMeansDiscretizer`.
        """
        self.clear()
        kwargs = {
            "std_per_bin": options.quantize.std_per_bin,
            "strategy": options.quantize.strategy,
            **kwargs,
        }
        self.discretizer = KMeansDiscretizer(**kwargs)
        self.lossy = lossy

    @property
    def bins(self) -> np.ndarray:
        """The most recent quantized array."""
        if self.is_empty:
            errmsg = "the quantizer is empty"
            raise ValueError(errmsg)
        return self._bins

    @property
    def inverse(self) -> np.ndarray:
        """The inverse mapping of the most recent quantized array."""
        if self.is_empty:
            errmsg = "the quantizer is empty"
            raise ValueError(errmsg)
        return self._inverse

    @property
    def counts(self) -> np.ndarray:
        """The counts of each unique bin in the most recent quantized array."""
        if self.is_empty:
            errmsg = "the quantizer is empty"
            raise ValueError(errmsg)
        return self._counts

    @property
    def is_empty(self) -> bool:
        """Check if the quantizer has been used."""
        return self._bins is None

    def quantize(self, X: np.ndarray) -> np.ndarray:
        """Quantize the input array.

        Parameters
        ----------
        X
            Input array to be quantized.
        """
        if not self.is_empty:
            errmsg = "the quantizer has already been used; call 'clear()' to reset"
            raise RuntimeError(errmsg)
        bins = self.discretizer.fit_transform(X)
        B, Inv, C = np.unique(bins, axis=0, return_inverse=True, return_counts=True)
        self._bins = B
        self._inverse = Inv
        self._counts = C
        if not self.lossy:
            self._array = X.copy()
        return self.discretizer.inverse_transform(B)

    def dequantize(
        self, X: np.ndarray | None = None, *, clear: bool = False
    ) -> np.ndarray:
        """Dequantize an array to the length and order of the mos quantized array.

        `X` may be whatever was returned by `quantize()`, or computed on its output
        while retaining the same length of the first axis.

        Parameters
        ----------
        X
            Input array to be dequantized.
            If `None`, then the original array is dequantized.
        clear
            Set to `True` to clear the quantizer state after dequantization.
        """
        if self.is_empty:
            errmsg = "the quantizer is empty; call 'quantize()' first"
            raise RuntimeError(errmsg)
        if X is None:
            if not self.lossy:
                dequantized = self._array.copy()
                if clear:
                    self.clear()
                return dequantized
            X = self.discretizer.inverse_transform(self.bins)
        if len(X) != len(self.bins):
            errmsg = "input array length does not match the quantizer state"
            raise ValueError(errmsg)
        dequantized = X[self.inverse]
        if clear:
            self.clear()
        return dequantized

    def clear(self) -> Self:
        """Clear the quantizer state."""
        self._bins = None
        self._inverse = None
        self._counts = None
        self._array = None
        return self

    def set_params(self, **kwargs: Any) -> Self:
        """Set parameters of the discretizer.

        `**kwargs` are passed to the `AdaptiveBinsDiscretizer`.
        """
        if (loss := kwargs.pop("lossy", None)) is not None:
            self.lossy = bool(loss)
        self.discretizer.set_params(**kwargs)
        return self
