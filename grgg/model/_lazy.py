import math
from collections.abc import Callable
from types import EllipsisType
from typing import Any

import jax.numpy as np
from flax import nnx


class LazyBroadcast:
    """Lazy broadcasting of arrays.

    Attributes
    ----------
    arrays
        Arrays to broadcast.
    op
        Broadcastable operation.

    Examples
    --------
    >>> x = np.array([1, 2, 3])
    >>> y = np.array([[10], [20], [30]])
    >>> lazy = LazyBroadcast(x, y, op=np.add)
    >>> lazy
    <LazyBroadcast (add), dtype=..., shape=(3, 3)>
    >>> lazy[:, :]
    Array([[11, 12, 13],
           [21, 22, 23],
           [31, 32, 33]], dtype=...)
    >>> lazy[0, :]
    Array([[11],
           [21],
           [31]], dtype=...)
    >>> lazy[:, 0]
    Array([11, 12, 13], dtype=...)
    >>> lazy[0, 1]
    Array(21, dtype=...)
    """

    def __init__(
        self, *arrays: np.ndarray, op: Callable[[np.ndarray, ...], np.ndarray]
    ) -> None:
        if not arrays:
            errmsg = "At least one array must be provided."
            raise ValueError(errmsg)
        arrays = tuple(np.asarray(arr) for arr in arrays)
        dtype = arrays[0].dtype
        for arr in arrays[1:]:
            dtype = np.promote_types(dtype, arr.dtype)
        self.arrays = tuple(arr.astype(dtype) for arr in arrays)
        self.op = nnx.jit(op)

    def __repr__(self) -> str:
        cn = self.__class__.__name__
        return f"<{cn} ({self.op.__name__}), dtype={self.dtype}, shape={self.shape}>"

    def __getitem__(self, args: Any) -> np.ndarray:
        if self.is_scalar:
            return self.op(*self.arrays)
        if args is Ellipsis:
            args = (slice(None),) * len(self.arrays)
        if not isinstance(args, tuple):
            args = (args,)
        if len(args) > len(self.arrays):
            errmsg = "too many indices"
            raise IndexError(errmsg)
        try:
            pos = list(args).index(Ellipsis)
            left, right = args[:pos], args[pos + 1 :]
            elen = len(self.arrays) - (len(left) + len(right))
            middle = (slice(None),) * elen
            args = left + middle + right
        except ValueError:
            pass
        if len(args) < len(self.arrays):
            n_missing = len(self.arrays) - len(args)
            args = args + (slice(None),) * n_missing
        args = [
            a if isinstance(a, slice | EllipsisType) else np.asarray(a) for a in args
        ]
        arrays = [a[i] for a, i in zip(self.arrays, args, strict=True)]
        out = self.op(*arrays)
        if out.size == 1 and all(
            np.isscalar(a) and not isinstance(a, slice) for a in args
        ):
            out = out.squeeze()
        return out

    @property
    def dtype(self) -> np.dtype:
        """Data type of the resulting array."""
        return self.arrays[0].dtype

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the resulting array."""
        return np.broadcast_shapes(*(a.shape for a in self.arrays))

    @property
    def ndim(self) -> int:
        """Number of dimensions of the resulting array."""
        return len(self.shape)

    @property
    def size(self) -> int:
        """Size of the resulting array."""
        if self.shape:
            return math.prod(self.shape)
        return 1

    @property
    def is_scalar(self) -> bool:
        """Whether the resulting array is a scalar."""
        return not self.shape


class LazyOuter(LazyBroadcast):
    """Lazy outer operation.

    Attributes
    ----------
    x
        First vector.
    y
        Second vector.
    op
        Binary operation to apply element-wise.

    Examples
    --------
    Lazy outer sum
    >>> outer = LazyOuter(np.array([1, 2]), np.array([10, 20]), op=np.add)
    >>> outer
    <LazyOuter (add), dtype=..., shape=(2, 2)>
    >>> outer[:, :]
    Array([[11, 21],
           [12, 22]], ...)
    >>> bool(np.array_equal(outer[:, :], outer[...]))
    True
    >>> outer[0, :]
    Array([11, 21], ...)
    >>> outer[:, 0]
    Array([[11],
           [12]], ...)
    >>> outer[0, 1]
    Array(21, ...)
    >>> outer[[0], 1]
    Array([[21]], ...)
    >>> outer[0, [1]]
    Array([21], ...)
    >>> outer[[0], [1]]
    Array([[21]], dtype=int32)
    >>> outer[[0, 1], [1]]
    Array([[21],
           [22]], ...)
    >>> outer[[1], [0, 1]]
    Array([[12, 22]], ...)

    A condensed pairwise vector can be obtained using the following syntax:
    >>> outer[:, ...]
    Array([21], ...)
    >>> outer[[0, 1], ...]
    Array([21], ...)
    >>> outer[1, ...]
    Array([], ...)

    Scalars can also be represented. In this case indexing is ignored.
    >>> outer = LazyOuter(2, 2)
    >>> outer[...]
    Array(4, ...)
    >>> outer[:, :]
    Array(4, ...)
    >>> outer[0, 0]
    Array(4, ...)
    >>> outer[[7], [0]]
    Array(4, ...)
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray | None = None,
        op: Callable[[np.ndarray, np.ndarray], np.ndarray] = np.multiply,
    ) -> None:
        x = np.asarray(x)
        y = x if y is None else np.asarray(y)
        if x.ndim > 1 or y.ndim > 1:
            errmsg = "Both inputs must be one-dimensional arrays."
            raise ValueError(errmsg)
        if not np.isscalar(x):
            x = x[:, None]
        super().__init__(x, y, op=op)

    def __getitem__(self, args: Any) -> np.ndarray:
        if isinstance(args, tuple) and len(args) == 2 and args[1] is Ellipsis:
            return self.__getitem_condensed(args[0])
        return super().__getitem__(args)

    def __getitem_condensed(self, i: slice | np.ndarray) -> np.ndarray:
        if isinstance(i, slice):
            i = range(i.start or 0, i.stop or self.shape[0], i.step or 1)  # type: ignore
        else:
            i = np.atleast_1d(np.asarray(i))
        i, j = np.triu_indices(len(i), k=1)
        x, y = self.arrays
        x = x[:, 0] if x.ndim == 2 else x
        return self.op(x[i], y[j])
