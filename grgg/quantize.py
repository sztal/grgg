from collections.abc import Iterable
from typing import Any, Self

import numpy as np

from . import options
from .utils.discretizers import KMeansDiscretizer

__all__ = ("ArrayQuantizer",)


class ArrayQuantizer:
    """Quantizer for numerical arrays.

    Currently only supports 2D arrays.

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
    Quantization is used to represent a larger array by a smaller
    array with rows representing clusters of similar rows in the original array.
    >>> rng = np.random.default_rng(412)
    >>> X = np.asarray([[0,0]]*3 + [[2,2]]*4 + [[4,4]]*2 + [[6,6]]*5)
    >>> rng.shuffle(X)
    >>> quantizer = ArrayQuantizer(std_per_bin=0.3)
    >>> X_quantized = quantizer.quantize(X)
    >>> X_quantized
    array([[0., 0.],
           [2., 2.],
           [4., 4.],
           [6., 6.]])

    In the lossless case, the quantizer just stores the copy of the original array,
    so whenever necessary it can be exactly reconstructed.
    >>> X_dequantized = quantizer.dequantize()
    >>> np.array_equal(X, X_dequantized)
    True

    However, in the simple case considered here, even the lossy quantization
    is exact.
    >>> quantizer = ArrayQuantizer(lossy=True, std_per_bin=0.3)
    >>> X_quantized = quantizer.quantize(X)
    >>> X_dequantized = quantizer.dequantize()
    >>> bool(np.array_equal(X, X_dequantized))
    True

    But in the general case, the dequantized array may differ from the original one.
    >>> X = rng.standard_normal((10, 2))
    >>> quantizer = ArrayQuantizer(lossy=True, std_per_bin=0.3)
    >>> X_quantized = quantizer.quantize(X)
    >>> X_dequantized = quantizer.dequantize()
    >>> bool(np.array_equal(X, X_dequantized))
    False

    Dequantized array is obtained by permuting the rows of the quantized
    array based on the inverse mapping defined by the quantizer.
    >>> X_quantized_inv = X_quantized[quantizer.inverse]
    >>> bool(np.array_equal(X_dequantized, X_quantized_inv))
    True

    The quantizer can also dequantize computations performed on the quantized array.
    >>> U = quantizer.dequantize(X_quantized * 2)
    >>> U_dequantized = X_dequantized * 2
    >>> bool(np.array_equal(U_dequantized, U))
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
        self.check_if_ready()
        return self._bins

    @property
    def inverse(self) -> np.ndarray:
        """The inverse mapping of the most recent quantized array."""
        self.check_if_ready()
        return self._inverse

    @property
    def counts(self) -> np.ndarray:
        """The counts of each unique bin in the most recent quantized array."""
        self.check_if_ready()
        return self._counts

    @property
    def is_empty(self) -> bool:
        """Check if the quantizer has been used."""
        return self._bins is None

    def map_ids(self, ids: int | Iterable) -> np.ndarray:
        """Remap a list of indices according to the most recent quantization.

        Parameters
        ----------
        ids
            Iterable of indices to be remapped.

        Examples
        --------
        >>> X = [[0, 0]]*3 + [[2, 2]]*4
        >>> quantizer = ArrayQuantizer()
        >>> quantizer.quantize(X)
        array([[0., 0.],
               [2., 2.]])
        >>> int(quantizer.map_ids(1))
        0
        >>> int(quantizer.map_ids(len(X)-1))
        1
        >>> quantizer.map_ids([0, 1, 2, 3])
        array([0, 0, 0, 1])
        """
        self.check_if_ready()
        if isinstance(ids, Iterable):
            ids = np.asarray(ids)
        return self.inverse[ids]

    def invmap_ids(self, ids: int | Iterable) -> np.ndarray:
        """Inverse remap a list of indices according to the most recent quantization.

        Parameters
        ----------
        ids
            Iterable of indices to be inverse remapped.

        Examples
        --------
        >>> rng = np.random.default_rng(17)
        >>> X = np.array([[0,0]]*3 + [[2,2]]*4 + [[4,4]]*2 + [[6,6]] * 5)
        >>> rng.shuffle(X)
        >>> quantizer = ArrayQuantizer()
        >>> X_quantized = quantizer.quantize(X)
        >>> X_dequantized = quantizer.dequantize()
        >>> X_quantized
        array([[0., 0.],
               [2., 2.],
               [4., 4.],
               [6., 6.]])
        >>> quantizer.invmap_ids(3)
        array([ 0,  2,  4,  9, 11])

        For inverting a single id, the folowing invariant holds:
        >>> id = 2  # example
        >>> bool(len(quantizer.invmap_ids(id)) == quantizer.counts[id])
        True

        >>> quantizer.invmap_ids([0, 1, 2])
        array([ 1,  3,  5,  6,  7,  8, 10, 12, 13])
        """
        self.check_if_ready()
        return np.where(np.isin(self.inverse, ids))[0]

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode to bin indices."""
        codes = self.discretizer.transform(X)
        return codes

    def quantize(self, X: np.ndarray) -> np.ndarray:
        """Quantize the input array.

        Quantization is guaranteed to be sort-stable, i.e. the order of the rows
        in the quantized array corresponds to the order of their first appearance
        of a representative of a given quantization bin in the original array.

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
        B, Ind, Inv, C = np.unique(
            bins, axis=0, return_index=True, return_inverse=True, return_counts=True
        )
        self._bins = B
        self._index = Ind
        self._inverse = Inv
        self._counts = C
        if not self.lossy:
            self._array = X.copy()
        return self.discretizer.inverse_transform(B)

    def dequantize(
        self,
        X: np.ndarray | None = None,
        i: int | Iterable | None = None,
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
        i
            Indices of rows in the quantized array to be dequantized.
            It is assumed that the quantized `X` is on order implied by `i`.
            If `None`, then the full dequantized array is returned.
        clear
            Set to `True` to clear the quantizer state after dequantization.

        Examples
        --------
        >>> X = np.array([[0, 0]]*3 + [[2, 2]]*4 + [[4, 4]]*2 + [[6, 6]] * 5)
        >>> quantizer = ArrayQuantizer(lossy=True)
        >>> X_quantized = quantizer.quantize(X)

        In this simple case even the lossy quantization is exact.
        >>> X_dequantized = quantizer.dequantize()
        >>> bool(np.array_equal(X_dequantized, X))
        True

        Dequantization is implemented as an inversion of the index mapping.
        >>> Inv = X_quantized[quantizer.inverse]
        >>> bool(np.array_equal(X_dequantized, Inv))
        True

        Quantizer can be also used to dequantize computations performed
        on the quantized array:
        >>> U = quantizer.dequantize(X_quantized * 2)
        >>> U_dequantized = X_dequantized * 2
        >>> bool(np.array_equal(U_dequantized, U))
        True

        Computations performed on a subset of the quantized array
        can be also dequantized by passing the corresponding indices:
        >>> qids = [1, 3]  # row ids in the quantized array
        >>> U = quantizer.dequantize(X_quantized[qids] * 5, qids)

        Is equivalent to:
        >>> ids = quantizer.invmap_ids(qids) # get row ids in the original array
        >>> U_dequantized = (X_dequantized * 5)[ids]
        >>> bool(np.array_equal(U_dequantized, U))
        True

        Check that in this case it matches with the result on the original array.
        >>> U_original = (X * 5)[ids]
        >>> bool(np.array_equal(U_original, U))
        True
        """
        self.check_if_ready()
        if X is None:
            if self.lossy:
                X = self.discretizer.inverse_transform(self.bins)
            else:
                X = self._array.copy()
        if X.ndim != 2:
            errmsg = "only 2D arrays are supported"
            raise ValueError(errmsg)
        if i is None:
            if self.lossy:
                if len(X) != len(self.bins):
                    errmsg = "array length does not match the number of quantizer bins"
                    raise ValueError(errmsg)
                ids = self.inverse
            else:
                ids = slice(None)
        else:
            if len(X) != (ilen := len(i) if isinstance(i, Iterable) else 1):
                errmsg = (
                    "array length does not match the number of provided indices "
                    f"({len(X)} != {ilen})"
                )
                raise ValueError(errmsg)
            idmap = {v: i for i, v in enumerate(dict.fromkeys(i))}
            ids = self.inverse[self.invmap_ids(i)]
            ids = np.array([idmap[v] for v in ids])
        dequantized = X[ids]  # type: ignore[index]
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

    def check_if_ready(self) -> None:
        if self.is_empty:
            errmsg = "the quantizer is empty; call 'quantize()' first"
            raise RuntimeError(errmsg)


__test__ = {
    "ArrayQuantizer.map_ids": ArrayQuantizer.map_ids.__doc__,
    "ArrayQuantizer.invmap_ids": ArrayQuantizer.invmap_ids.__doc__,
}
