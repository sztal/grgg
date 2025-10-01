import math
from dataclasses import replace
from typing import Any, Self

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.cluster.vq import vq
from sklearn.cluster import BisectingKMeans

from grgg._typing import IntVector, Reals
from grgg.abc.modules import AbstractModule
from grgg.utils.misc import split_by


@jax.jit
def _reshape_in(data: jnp.ndarray) -> jnp.ndarray:
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim > 2:
        data = data.reshape(len(data), -1)
    return data


@jax.jit
def _assign_codes(states: jnp.ndarray, codebook: jnp.ndarray):
    states = _reshape_in(states)
    codes, _ = vq(states, codebook)
    return codes


class ArrayQuantizer(AbstractModule):
    """Quantizer for numerical arrays.

    Attributes
    ----------
    codebook
        Quantization codebook (k-means centroids).
    codes
        Assigned quantization codes for the original data.
    inverse
        Inverse mapping from codes to original indices.

    Examples
    --------
    Below we quantize a 3D array into using 32 k-means centroids.
    >>> import jax.numpy as jnp
    >>> data = jnp.arange(400).reshape(100, 2, 2)
    >>> quantizer = ArrayQuantizer.from_data(data, n_codes=32, random_state=17)

    >>> quantizer
    <ArrayQuantizer (100, 2, 2)->(32, 2, 2) at ...>
    >>> quantizer.codebook.shape
    (32, 4)
    >>> quantizer.codes.shape
    (100,)

    Once created, the quantizer can be used to co-quantize new data
    together with any data of the same dimensionality as the original data.
    >>> new_data = jnp.arange(200).reshape(100, 2, 1)
    >>> quantized, quantized_new = quantizer.quantize(data, new_data)
    >>> quantized.shape
    (32, 2, 2)
    >>> quantized_new.shape
    (32, 2, 1)

    One can verify that the correlation between the original data
    and the new data being a linear function of the original data
    is preserved upon quantization.
    >>> def f(x): return x * 1.34 + 2.1
    >>> X = data
    >>> Y = f(X)
    >>> jnp.corrcoef(X.ravel(), Y.ravel())[0, 1].item()
    1.0
    >>> qX, qY = quantizer.quantize(X, Y)
    >>> jnp.corrcoef(qX.ravel(), qY.ravel())[0, 1].item()
    1.0
    """

    codebook: Reals = eqx.field(converter=jnp.asarray, repr=False)
    codes: IntVector = eqx.field(converter=jnp.asarray, repr=False)
    inverse: tuple[IntVector] = eqx.field(converter=tuple, repr=False)
    trailing_shape: tuple[int, ...] = eqx.field(repr=False, static=True)

    def __repr__(self) -> str:
        qshape = (self.n_codes, *self.shape[1:])
        return f"<{self.__class__.__name__} {self.shape}->{qshape} at {hex(id(self))}>"

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the original data."""
        n = len(self.codes)
        return (n, *self.trailing_shape)

    @property
    def ndim(self) -> int:
        """Number of dimensions of the original data."""
        return 1 + len(self.shape)

    @property
    def size(self) -> int:
        """Total number of elements in the original data."""
        return math.prod(self.shape)

    @property
    def n_codes(self) -> int:
        """Number of quantization codes."""
        return len(self.codebook)

    @property
    def counts(self) -> IntVector:
        """Counts of each quantization code in the original data."""
        return jnp.array([len(idx) for idx in self.inverse], dtype=int)

    @property
    def dequantize(self) -> "Dequantization":
        """Dequantization handler."""
        return Dequantization(self)

    @classmethod
    def from_data(cls, data: jnp.ndarray, n_codes: int, **kwargs: Any) -> Self:
        """Create a quantizer from data by k-means clustering.

        Parameters
        ----------
        data
            Data to be quantized.
        n_codes
            Number of quantization codes.
        **kwargs
            Additional keyword arguments passed to
            :class:`sklearn.cluster.BisectingKMeans`.
        """
        data = cls._validate_data(data)
        shape = data.shape
        reshaped_data = cls.reshape_in(data)
        kwargs = {
            "init": "k-means++",
            "algorithm": "lloyd",
            "bisecting_strategy": "biggest_inertia",
            **kwargs,
        }
        if not jnp.issubdtype(reshaped_data.dtype, jnp.floating):
            reshaped_data = reshaped_data.astype(float)
        _input = np.asarray(reshaped_data)
        kmeans = BisectingKMeans(n_clusters=n_codes, **kwargs).fit(_input)
        codebook = jnp.asarray(kmeans.cluster_centers_)
        codes = jnp.asarray(kmeans.labels_)
        order = jnp.argsort(codes)
        index = jnp.arange(len(codes))[order]
        inverse = split_by(index, codes[order])
        return cls(codebook, codes, inverse, shape[1:])

    def equals(self, other: object) -> bool:
        return (
            super().equals(other)
            and jnp.array_equal(self.codebook, other.codebook)
            and all(
                jnp.array_equal(a, b)
                for a, b in zip(self.inverse, other.inverse, strict=True)
            )
            and self.shape == other.shape
        )

    @staticmethod
    def reshape_in(data: jnp.ndarray) -> jnp.ndarray:
        """Reshape possibly multi-dimensional data into 2D array."""
        return _reshape_in(data)

    def reshape_out(
        self, data: jnp.ndarray, trailing_shape: tuple[int, ...] | None = None
    ) -> jnp.ndarray:
        """Reshape 2D quantized data to have the original trailing shape."""
        if trailing_shape is None:
            trailing_shape = self.trailing_shape
        return data.reshape((-1, *trailing_shape))

    def quantize(
        self,
        states: jnp.ndarray,
        *data: jnp.ndarray,
    ) -> jnp.ndarray | tuple[jnp.ndarray, ...]:
        """Quantize the provided data.

        Parameters
        ----------
        states
            States to be quantized.
        *data
            Additional data arrays to be co-quantized based on `states`.
            Must have the same first dimension as `states`.

        Returns
        -------
        quantized, *quantized data
            Quantized states corresponding to the provided data.
            Additional quantized arrays are returned only if additional
            data arrays are provided, so if they were not provided the return
            value is not a tuple.
        """
        if any(len(arr) != len(states) for arr in data):
            errmsg = "all data arrays must have the same first dimension."
            raise ValueError(errmsg)
        states = self._validate_data(states)
        codes = self.assign_codes(states)
        uniq_codes, index = jnp.unique(codes, return_index=True)
        uniq_codes = uniq_codes[jnp.argsort(index)]
        quantized = self.codebook[uniq_codes]
        quantized = self.reshape_out(quantized, states.shape[1:])
        if not data:
            return quantized
        quantized_data = []
        for array in data:
            array = self._validate_data(array)
            groups = split_by(array, codes)
            qarray = jnp.stack([g.mean(0) for g in groups])[uniq_codes]
            quantized_data.append(qarray)
        return (quantized, *quantized_data)

    def assign_codes(self, states: jnp.ndarray) -> IntVector:
        """Assign quantization codes to the provided states."""
        return _assign_codes(states, self.codebook)

    @staticmethod
    def _validate_data(data: jnp.ndarray) -> jnp.ndarray:
        return jnp.atleast_1d(jnp.asarray(data))


class Dequantization(eqx.Module):
    """Dequantization process handler.

    Attributes
    ----------
    quantizer
        The associated quantizer.
    indices
        Optional indices of the target rows in the original data to be dequantized.
        If not specified, all rows matching the quantized states are returned.

    Examples
    --------
    Below we will use quantized states to reconstruct the original data
    and co-dequantize new data.
    >>> import jax.numpy as jnp
    >>> X = jnp.arange(400).reshape(100, 2, 2)
    >>> quantizer = ArrayQuantizer.from_data(X, n_codes=32, random_state=17)
    >>> quantizer.dequantize
    <Dequantization of ArrayQuantizer (100, 2, 2)->(32, 2, 2) at ...>
    >>> def f(x): return x * 1.34 + 2.1
    >>> def error(x, y): return float(jnp.linalg.norm(x - y) / jnp.linalg.norm(x))
    >>> Y = f(X)
    >>> qX, qY = quantizer.quantize(X, Y)
    >>> dX, dY = quantizer.dequantize(qX, qY)
    >>> error(X, dX)
    0.016324782
    >>> error(Y, dY)
    0.016388867

    The alignment between the original and dequantized data can be also
    measured by correlation.
    >>> def corr(x, y): return jnp.corrcoef(x.ravel(), y.ravel())[0, 1].item()
    >>> corr(X, dX)
    0.99946892
    >>> corr(Y, dY)
    0.99945795

    One can also dequantize only a subset of the original data.
    Here we will dequantize an arbitrary selection of rows of the original data
    from the same full set of quantized states.
    >>> idx = jnp.array([3, 17, 10, 90, 56, 90, 54, 31, 0, 2, 80, 34, ])
    >>> dX, dY = quantizer.dequantize[idx](qX, qY)
    >>> error(X[idx], dX)
    0.022174103
    >>> corr(X[idx], dX)
    0.999712407
    >>> error(Y[idx], dY)
    0.021866155
    >>> corr(Y[idx], dY)
    0.999696373
    """

    quantizer: ArrayQuantizer
    indices: IntVector | None = eqx.field(default=None, repr=False)

    def __repr__(self) -> str:
        cn = self.__class__.__name__
        return f"<{cn} of {repr(self.quantizer)[1:-1]}>"

    def __getitem__(self, indices: Any) -> Self:
        if indices is Ellipsis:
            indices = slice(None)
        if isinstance(indices, tuple):
            if len(indices) != 1:
                errmsg = "Only 1D indexing is supported."
                raise IndexError(errmsg)
            indices = indices[0]
        indices = self._validate_indices(indices)
        return replace(self, indices=indices)

    def __call__(
        self, states: jnp.ndarray, *data: jnp.ndarray
    ) -> jnp.ndarray | tuple[jnp.ndarray, ...]:
        """Dequantize the provided data.

        Parameters
        ----------
        states
            Quantized states to be dequantized.
            Must map to unique quantization codes.
        *data
            Additional data arrays to be co-dequantized based on `states`.
            Must have the same first dimension as `states`.

        Returns
        -------
        dequantized, *dequantized data
            Dequantized states corresponding to the provided data.
            Additional dequantized arrays are returned only if additional
            data arrays are provided, so if they were not provided the return
            value is not a tuple.
        """
        if any(len(arr) != len(states) for arr in data):
            errmsg = "All data arrays must have the same first dimension."
            raise ValueError(errmsg)
        states = self.quantizer._validate_data(states)
        codes = self.quantizer.assign_codes(states)
        if len(jnp.unique(codes)) < len(codes):
            err = "multiple states correspond to the same quantized codes"
            raise ValueError(err)
        return self._dequantize(self.indices, states, codes, *data)

    def _dequantize(
        self,
        indices: jnp.ndarray | None,
        states: jnp.ndarray,
        codes: jnp.ndarray,
        *data: jnp.ndarray,
    ) -> jnp.ndarray | tuple[jnp.ndarray, ...]:
        if indices is None:
            indices = jnp.arange(self.quantizer.shape[0])
        target_codes = self.quantizer.codes[indices]
        dequantized = self.quantizer.codebook[target_codes]
        dequantized = self.quantizer.reshape_out(dequantized, states.shape[1:])
        if not data:
            return dequantized
        codes_order = jnp.argsort(codes)
        dequantized_data = []
        for array in data:
            array = self.quantizer._validate_data(array)[codes_order][target_codes]
            dequantized_data.append(array)
        return (dequantized, *dequantized_data)

    def _validate_indices(self, indices: jnp.ndarray | None) -> IntVector:
        """Validate the provided indices for (de)quantization."""
        in_units = self.quantizer.shape[0]
        if indices is None:
            return jnp.arange(in_units)
        if isinstance(indices, slice):
            indices = jnp.arange(*indices.indices(in_units))
        indices = jnp.asarray(indices)
        if indices.ndim > 1:
            errmsg = "Only 1D indexing is supported."
            raise IndexError(errmsg)
        if jnp.issubdtype(indices.dtype, jnp.bool_):
            indices = jnp.where(indices)[0]
        if not jnp.issubdtype(indices.dtype, jnp.integer):
            errmsg = "Indices must be integers or boolean."
            raise TypeError(errmsg)
        if indices.min() < 0 or indices.max() >= in_units:
            errmsg = f"Indices must be in the range [0, {in_units})."
            raise IndexError(errmsg)
        return indices
