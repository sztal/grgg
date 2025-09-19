import math
from collections.abc import Callable
from dataclasses import replace
from typing import Any, Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import DTypeLike

from grgg._typing import Integers, Vector

from .indexing import CartesianCoordinates, IndexArg

AijCallableT = Callable[[IndexArg | tuple[IndexArg, ...]], jnp.ndarray]


class LazyArray(eqx.Module):
    """Lazy array wrapper around a callable.

    Attributes
    ----------
    shape
        The shape of the array.
    a_ij
        A callable that takes a Numpy-style index and returns the corresponding
        subarray. The function should be jittable and vectorized over the index.
        Alternatively, a subclass can define `a_ij` as a method.
    dtype
        The data type of the array.
    coords
        A :class:`CartesianCoordinates` instance for the array.
        It is used to translate between Numpy-style indices and Cartesian coordinates
        relative to the array shape.

    Examples
    --------
    Here we create a simple lazy 3D outer sum.
    >>> import jax.numpy as jnp
    >>> from grgg.lazy import LazyArray
    >>> x1 = jnp.arange(10)
    >>> x2 = jnp.arange(8)
    >>> x3 = jnp.arange(5)
    >>> # Target array
    >>> X = x1[:, None, None] + x2[None, :, None] + x3[None, None, :]
    >>> X.shape
    (10, 8, 5)

    Now we create a lazy array that computes the same thing.
    >>> def a_ij(i, j, k):
    ...     return x1[i] + x2[j] + x3[k]
    >>>
    >>> A = LazyArray(X.shape, a_ij, dtype=X.dtype)
    >>> A
    <LazyArray (a_ij), shape=(10, 8, 5), dtype=...>
    >>> jnp.array_equal(A[...], X).item()
    True
    >>> jnp.array_equal(A[3], X[3]).item()
    True
    >>> jnp.array_equal(A[2:5, 1:4], X[2:5, 1:4]).item()
    True
    >>> jnp.array_equal(A[[1, 3, 5], [0, 2, 4]], X[[1, 3, 5], [0, 2, 4]]).item()
    True
    >>> i = jnp.array([0, 2, 4])
    >>> j = jnp.array([1, 3])
    >>> k = jnp.array([0, 2, 4])
    >>> idx = jnp.ix_(i, j, k)
    >>> jnp.array_equal(A[idx], X[idx]).item()
    True
    >>> mask = jnp.zeros(X.shape, dtype=bool)
    >>> jnp.array_equal(A[mask], X[mask]).item()
    True
    >>> idx = A.coords.i_[2:5, [0, 2, -3], [1, 3, 3]]
    >>> jnp.array_equal(A[idx], X[idx]).item()
    True
    >>> mask = [False]*1+[True]*3+[False]*4
    >>> jnp.array_equal(A[:, mask, :, None], X[:, mask, :, None]).item()
    True
    """

    shape: tuple[int, ...] = eqx.field(static=True)
    f: AijCallableT = eqx.field(static=True)
    dtype: DTypeLike | None = eqx.field(static=True)
    coords: CartesianCoordinates = eqx.field()

    def __init__(
        self,
        shape: tuple[int, ...],
        f: AijCallableT | None = None,
        *,
        dtype: DTypeLike | None = None,
        coords: CartesianCoordinates | None = None,
    ) -> None:
        self.shape = (
            (int(shape),) if jnp.isscalar(shape) else tuple(int(s) for s in shape)
        )
        if hasattr(self, "a_ij") and f is not None:
            cn = self.__class__.__name__
            errmsg = (
                f"it appears that {cn} already defines `a_ij`, "
                "so `f` argument should not be provided"
            )
            raise ValueError(errmsg)
        if f is not None:
            f = jax.jit(f)
        elif not hasattr(self, "a_ij"):
            errmsg = "`a_ij` must be provided either as an argument `f` or a method"
            raise ValueError(errmsg)
        self.f = f
        self.dtype = dtype
        self.coords = CartesianCoordinates(shape) if coords is None else coords

    def __check_init__(self) -> None:
        if getattr(self, "_f", None) is None and not hasattr(self, "a_ij"):
            errmsg = "`a_ij` must be provided either as an argument `f` or a method"
            raise ValueError(errmsg)
        if self.shape != self.coords.shape:
            errmsg = "`coords` shape must match `shape`"
            raise ValueError(errmsg)

    def __repr__(self) -> str:
        cn = self.__class__.__name__
        attrs = f"shape={self.shape}"
        if self.dtype is not None:
            attrs += f", dtype={self.dtype}"
        return f"<{cn} ({self._f.__name__}), {attrs}>"

    def __getitem__(self, args: Any) -> jnp.ndarray:
        target_shape = None
        if not isinstance(args, tuple):
            args = (args,)
        if any(a is None for a in args):
            target_shape = self.coords.s_[args]
            args = tuple(a for a in args if a is not None)
        coords = self.coords[args]
        out = self._f(*coords)
        if self.dtype is not None and out.dtype != self.dtype:
            out = out.astype(self.dtype)
        if target_shape is not None:
            out = out.reshape(target_shape)
        return out

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self.shape)

    @property
    def size(self) -> int:
        """Total number of elements."""
        return math.prod(self.shape)

    @property
    def _f(self) -> AijCallableT:
        """The callable that takes a Numpy-style index and returns the corresponding
        subarray.
        """
        f = self.f if self.f is not None else self.a_ij
        if f is None:
            errmsg = "`a_ij` must be provided either as an argument `f` or a method"
            raise AttributeError(errmsg)
        return f

    def astype(self, dtype: DTypeLike) -> Self:
        """Return a new instance with the given dtype."""
        return replace(self, dtype=dtype)

    def reshape(self, *shape: int) -> Self:
        """Return a new instance with the given shape."""
        if not shape:
            errmsg = "shape must be non-empty"
            raise ValueError(errmsg)
        if isinstance(shape[0], tuple):
            if any(isinstance(s, tuple) for s in shape[1:]):
                errmsg = "only one tuple can be provided as shape"
                raise ValueError(errmsg)
            return self.reshape(*shape)
        if math.prod(shape) != self.size:
            errmsg = f"cannot reshape array of size {self.size} into shape {shape}"
            raise ValueError(errmsg)
        return replace(self, shape=shape, coords=CartesianCoordinates(shape))


class LazyView(LazyArray):
    """Lazy view of an array, with optional element-wisze transformation.

    Attributes
    ----------
    x
        The input array.
    op
        An optional element-wise operation to apply to the input array.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from grgg.lazy import LazyView
    >>> x = jnp.arange(10)
    >>> V = LazyView(x, jnp.square)
    >>> V
    <LazyView (square), shape=(10,), dtype=...>
    >>> jnp.array_equal(V[...], jnp.square(x)).item()
    True

    Scalar case.
    >>> V = LazyView(3)
    >>> V
    <LazyView shape=(), dtype=...>
    >>> V[...].item()
    3
    >>> V[:10000].item()
    3
    """

    x: jnp.ndarray
    op: Callable[[jnp.ndarray], jnp.ndarray] | None = eqx.field(static=True)

    def __init__(
        self, x: jnp.ndarray, op: Callable[[jnp.ndarray], jnp.ndarray] | None = None
    ) -> None:
        x = jnp.asarray(x)
        dtype = x.dtype
        self.x = x
        self.op = jax.jit(op) if op is not None else op
        super().__init__(x.shape, dtype=dtype)

    def __repr__(self) -> str:
        cn = self.__class__.__name__
        attrs = f"shape={self.shape}, dtype={self.dtype}"
        if self.op is not None:
            attrs = f"({self.op.__name__}), " + attrs
        return f"<{cn} {attrs}>"

    def __getitem__(self, args: Any) -> jnp.ndarray:
        if self.is_scalar:
            args = Ellipsis
        return super().__getitem__(args)

    def a_ij(self, i: Integers | None = None) -> jnp.ndarray:
        return self.op(self.x[i]) if self.op is not None else self.x[i]

    @property
    def is_scalar(self) -> bool:
        """Whether the view is a scalar."""
        return jnp.isscalar(self.x)


class LazyOuter(LazyArray):
    """Lazy outer operation on two 1D arrays.

    Attributes
    ----------
    x
        The first input array.
    y
        The second input array.
    op
        The outer operation.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from grgg.lazy import LazyOuter
    >>> x1 = jnp.array([1, 2, 3])
    >>> x2 = jnp.array([4, 5])
    >>> O = LazyOuter(x1, x2)
    >>> O
    <LazyOuter (multiply), shape=(3, 2), dtype=...>
    >>> jnp.array_equal(O[...], jnp.outer(x1, x2)).item()
    True

    Scalar case.
    >>> O = LazyOuter(3, 4)
    >>> O
    <LazyOuter (multiply), shape=(), dtype=...>
    >>> O[...].item()
    12

    The convenion is that in this case any indexing
    is allowed, but always produces the same scalar result.
    >>> O[:5, ..., [1,2]].item()
    12
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
        shape = () if self.is_scalar else (len(self.x), len(self.y))
        super().__init__(shape, dtype=dtype)

    def __repr__(self) -> str:
        cn = self.__class__.__name__
        return f"<{cn} ({self.op.__name__}), shape={self.shape}, dtype={self.dtype}>"

    def __getitem__(self, args: Any) -> jnp.ndarray:
        if self.is_scalar:
            args = Ellipsis
        return super().__getitem__(args)

    @property
    def is_scalar(self) -> bool:
        """Whether the outer product is a scalar."""
        return jnp.isscalar(self.x) and jnp.isscalar(self.y)

    def a_ij(self, i: Integers | None = None, j: Integers | None = None) -> jnp.ndarray:
        x = self.x if i is None else self.x[i]
        y = self.y if j is None else self.y[j]
        return self.op(x, y)
