from collections.abc import Iterable
from functools import singledispatchmethod
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
        If `False`, then a copy of the original array is stored,
        so it is returned exactly by `dequantize()`.
        Otherwise, the original array is reconstructed from the bins,
        which may introduce some information loss.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((10, 2))
    >>> quantizer = ArrayQuantizer(std_per_bin=0.3)
    >>> X_quantized = quantizer.quantize(X)
    >>> bool(np.array_equal(quantizer.dequantize(), X))
    True

    Quantizer can also remap indices according to the quantization:
    >>> Y = quantizer.dequantize(X_quantized)
    >>> idx = [1, 9]
    >>> bool(np.array_equal(Y[idx], X_quantized[quantizer.remap_ids(idx)]))
    True

    The same can be done directly using the 'dequantize' method:
    >>> bool(np.array_equal(Y[idx], quantizer.dequantize(X_quantized, idx)))
    True

    >>> idx = 7
    >>> bool(np.array_equal(Y[idx], quantizer.dequantize(X_quantized, idx)))
    True
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
        self._check_if_ready()
        return self._bins

    @property
    def inverse(self) -> np.ndarray:
        """The inverse mapping of the most recent quantized array."""
        self._check_if_ready()
        return self._inverse

    @property
    def counts(self) -> np.ndarray:
        """The counts of each unique bin in the most recent quantized array."""
        self._check_if_ready()
        return self._counts

    @property
    def is_empty(self) -> bool:
        """Check if the quantizer has been used."""
        return self._bins is None

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode to bin indices."""
        return self.discretizer.transform(X)

    @singledispatchmethod
    def remap_ids(self, ids: int | Iterable) -> np.ndarray:
        """Remap a list of indices according to the most recent quantization.

        Parameters
        ----------
        ids
            Iterable of indices to be remapped.
        """
        self._check_if_ready()
        if isinstance(ids, Iterable):
            ids = np.asarray(ids)
        return self.inverse[ids]

    @remap_ids.register
    def _(self, ids: slice) -> np.ndarray:
        ids = np.arange(ids.start, ids.stop, ids.step)
        return self.remap_ids(ids)

    def quantize(self, X: np.ndarray) -> np.ndarray:
        """Quantize the input array.

        Parameters
        ----------
        X
            Input array to be quantized.
        """
        if not self.is_empty:
            errmsg = "the quantizer is not empty; call 'clear()' first"
            raise RuntimeError(errmsg)
        self.discretizer.fit(X)
        bins = self.encode(X)
        B, Inv, C = np.unique(bins, axis=0, return_inverse=True, return_counts=True)
        self._bins = B
        self._inverse = Inv
        self._counts = C
        if not self.lossy:
            self._array = X.copy()
        return self.discretizer.inverse_transform(B)

    def dequantize(
        self,
        X: np.ndarray | None = None,
        idx: Iterable | slice | None = None,
        *,
        clear: bool = False,
    ) -> np.ndarray:
        """Dequantize an array to the length and order of the mos quantized array.

        `X` may be whatever was returned by `quantize()`, or computed on its output
        while retaining the same length of the first axis.

        Parameters
        ----------
        X
            Input array to be dequantized.
            If `None`, then the original array is dequantized.
        idx
            Optional indices used to align `X`.
            Must match the length of `X` if provided.
            Ignored if `X` is `None`.
        clear
            Set to `True` to clear the quantizer state after dequantization.
        """
        self._check_if_ready()
        if X is None:
            dequantized = (
                self.discretizer.inverse_transform(self.bins)
                if self.lossy
                else self._array.copy()
            )
        else:
            if idx is None:
                if len(X) != len(self.bins):
                    errmsg = "input array length does not match the quantizer state"
                    raise ValueError(errmsg)
                idx = self.inverse
            else:
                idx = self.remap_ids(idx)
            dequantized = X[idx]
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

        `**kwargs` are passed to the `KMeansDiscretizer`.
        """
        if (loss := kwargs.pop("lossy", None)) is not None:
            self.lossy = bool(loss)
        self.discretizer.set_params(**kwargs)
        return self

    def _check_if_ready(self) -> None:
        if self.is_empty:
            errmsg = "the quantizer is empty; call 'quantize()' first"
            raise RuntimeError(errmsg)
