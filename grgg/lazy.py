from collections.abc import Callable
from dataclasses import replace
from typing import Any, Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import DTypeLike

from grgg._typing import IntVector, Vector


class LazyOuter(eqx.Module):
    """Lazy outer operation on two 1D arrays.

    Only standard indexing with integers or broadcastable 1D integer arrays
    is supported.

    Attributes
    ----------
    x
        The first input array.
    y
        The second input array.
    op
        The outer operation.
        It must be vectorized over 1D arrays of the same shape.

    Examples
    --------
    >>> x1 = jnp.array([1, 2, 3])
    >>> x2 = jnp.array([4, 5])
    >>> B = LazyOuter(x1, x2)
    >>> B
    <LazyOuter (multiply), shape=(3, 2), dtype=...>
    >>> B[0, 1]
    Array(5, ...)
    >>> B[:2, 1]
    Array([[ 5],
           [10]], ...)
    >>> B[1, :]
    Array([ 8, 10], ...)
    >>> B[1:3, 0:1]
    Array([[ 8],
           [12]], ...)
    >>> B[...]
    Array([[ 4,  5],
           [ 8, 10],
           [12, 15]], ...)
    >>> B[jnp.array([0, 2]), jnp.array([1, 0])]
    Array([ 5, 12], dtype=int32)

    If indexed with a single item, the result is a new instance with a filtered `x`.
    This allows for selecting rectangular subsets using integer arrays.
    >>> B[[0, 2]][:, [0]]
    Array([[ 4],
           [12]], dtype=int32)

    If initialized with scalars, the result is a scalar regardless of indexing.
    >>> C = LazyOuter(2, 3)
    >>> C
    <LazyOuter (multiply), shape=(), dtype=...>
    >>> C[1, :4]
    Array(6, ...)
    """

    x: Vector
    y: Vector
    op: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]

    def __init__(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray | None = None,
        op: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = jnp.multiply,
    ) -> None:
        x = jnp.asarray(x)
        y = x if y is None else jnp.asarray(y)
        if x.ndim > 1 or y.ndim > 1:
            errmsg = "Both input arrays must be 1D."
            raise ValueError(errmsg)
        if not jnp.isscalar(x) or not jnp.isscalar(y):
            x = jnp.atleast_1d(x)
            y = jnp.atleast_1d(y)
        dtype = jnp.promote_types(x.dtype, y.dtype)
        self.x = x.astype(dtype)
        self.y = y.astype(dtype)
        self.op = jax.jit(op)

    def __repr__(self) -> str:
        cn = self.__class__.__name__
        return f"<{cn} ({self.op.__name__}), shape={self.shape}, dtype={self.dtype}>"

    def __getitem__(self, args: Any) -> jnp.ndarray | Self:
        if args is Ellipsis:
            if self.is_scalar:
                return self.op(self.x, self.y)
            return self[slice(self.shape[0]), slice(self.shape[1])]
        if not isinstance(args, tuple):
            args = (args,)
        if len(args) == 1:
            if not jnp.isscalar(self.x):
                return replace(self, x=self.x.flatten()[args[0],])
            return self
        if self.is_scalar:
            return self.op(self.x, self.y)
        if not args:
            return self[...]
        i1, i2 = self._process_args(args)
        if isinstance(i1, jnp.ndarray) and (
            jnp.isscalar(i1) or isinstance(i2, jnp.ndarray)
        ):
            return self.op(self.x[i1], self.y[i2])
        return self.op(self.x[i1, None], self.y[i2])

    @property
    def shape(self) -> tuple[int, int]:
        if self.is_scalar:
            return ()
        return (len(self.x), len(self.y))

    @property
    def dtype(self) -> DTypeLike:
        return self.x.dtype

    @property
    def size(self) -> int:
        return self.shape[0] * self.shape[1]

    @property
    def is_scalar(self) -> bool:
        return jnp.isscalar(self.x) and jnp.isscalar(self.y)

    def _process_args(
        self, args: tuple[Any, Any]
    ) -> tuple[IntVector | slice, IntVector | slice]:
        i1, i2 = args
        if i1 is Ellipsis and i2 is Ellipsis:
            errmsg = "ellipsis can only appear once in the index."
            raise IndexError(errmsg)
        if i1 is Ellipsis:
            i1 = slice(None)
        if i2 is Ellipsis:
            i2 = slice(None)
        i1 = self._process_arg(i1)
        i2 = self._process_arg(i2)
        if isinstance(i1, jnp.ndarray) and isinstance(i2, jnp.ndarray):
            jnp.broadcast_shapes(i1.shape, i2.shape)
        return i1, i2

    def _process_arg(self, arg: slice | jnp.ndarray) -> IntVector:
        if isinstance(arg, slice):
            return arg
        arg = jnp.asarray(arg)
        if arg.dtype == bool:
            errmsg = "boolean masks are not supported"
            raise IndexError(errmsg)
        if arg.ndim > 1:
            errmsg = "Only 1D arrays are supported for indexing"
            raise IndexError(errmsg)
        return arg
