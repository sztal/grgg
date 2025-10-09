from collections.abc import Callable
from dataclasses import replace
from typing import Any, ClassVar, Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import DTypeLike

from grgg._typing import Integers, RealVector

from .indexing import DynamicIndexExpression, IndexArgT, Shaped

IndexCallableT = Callable[[IndexArgT | tuple[IndexArgT, ...]], jnp.ndarray]


class LazyArray(Shaped):
    """Lazy array wrapper around a callable.

    Attributes
    ----------
    shape
        The shape of the array.
    f
        A callable that takes a Numpy-style index and returns the corresponding
        subarray. The function should be jittable and vectorized over the index.
        Alternatively, a subclass can define 'function' method.
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
    >>> from grgg.utils.lazy import LazyArray
    >>> from grgg.utils.indexing import IndexExpression
    >>> x1 = jnp.arange(10)
    >>> x2 = jnp.arange(8)
    >>> x3 = jnp.arange(5)
    >>> # Target array
    >>> X = x1[:, None, None] + x2[None, :, None] + x3[None, None, :]
    >>> X.shape
    (10, 8, 5)

    Now we create a lazy array that computes the same thing.
    >>> def function(i, j, k):
    ...     return x1[i] + x2[j] + x3[k]
    >>>
    >>> A = LazyArray(X.shape, function, dtype=X.dtype)
    >>> A
    <LazyArray (function) shape=(10, 8, 5) dtype=...>
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
    >>> expr = IndexExpression(A.shape)  # Utility for generating indexing arguments
    >>> idx = expr[2:5, [0, 2, -3], [1, 3, 3]]
    >>> jnp.array_equal(A[idx], X[idx]).item()
    True
    >>> mask = [False]*1+[True]*3+[False]*4
    >>> jnp.array_equal(A[:, mask, :, None], X[:, mask, :, None]).item()
    True

    Now, let us check that computations are jittable.
    It is recommended to use :func:`equinox.filter_jit` for JIT compilation
    of functions that take :class:`LazyArray` as input.
    >>> import equinox as eqx
    >>> @eqx.filter_jit
    ... def compute_sum(A, *args):
    ...     return jnp.sum(A[args])
    >>>
    >>> compute_sum(A, 2, 3)
    Array(35, ...)
    >>> compute_sum(A, 2, [0, 2])
    Array(50, ...)
    >>> compute_sum(A, [0, 2], [1, 3])
    Array(50, ...)
    >>> compute_sum(A, ...)
    Array(4000, ...)
    >>> compute_sum(A, ..., 1)
    Array(720, ...)
    >>> compute_sum(A, slice(2, 5), ..., [0, 2])
    Array(360, ...)
    >>> i = jnp.array([0, 2, 4])
    >>> j = jnp.array([1, 3])
    >>> k = jnp.array([0, 2, 4])
    >>> idx = jnp.ix_(i, j, k)
    >>> compute_sum(A, *idx)
    Array(108, ...)
    >>> compute_sum(A, *idx[:2], slice(1, 3))
    Array(66, ...)

    Boolean indexing is not supported in JIT-compiled functions,
    so we do not test it here.

    To allow for differentiation, some extra care is needed.
    Most importantly, it is best to work with a subclass of :class:`LazyArray`
    that stores the low rank factors as attributes.
    >>> class Lazy3DSum(LazyArray):
    ...     x1: jnp.ndarray
    ...     x2: jnp.ndarray
    ...     x3: jnp.ndarray
    ...
    ...     def __init__(self, x1, x2, x3):
    ...         self.x1 = x1
    ...         self.x2 = x2
    ...         self.x3 = x3
    ...         shape = (len(x1), len(x2), len(x3))
    ...         super().__init__(shape)
    ...
    ...     def function(self, i, j, k):
    ...         return self.x1[i] + self.x2[j] + self.x3[k]
    >>>
    >>> x1 = jnp.arange(10, dtype=jnp.float32)
    >>> x2 = jnp.arange(8, dtype=jnp.float32)
    >>> x3 = jnp.arange(5, dtype=jnp.float32)
    >>> A = Lazy3DSum(x1, x2, x3)
    >>>
    >>> @eqx.filter_jit
    ... def compute_sum(A, *args):
    ...     return jnp.sum(A[args])
    >>>
    >>> compute_sum(A, ...)
    Array(4000., ...)
    >>> compute_sum_grad = eqx.filter_grad(compute_sum)
    >>> g = compute_sum_grad(A, ...)
    >>> gx1_expected = jnp.ones_like(x1) * (A.shape[1] * A.shape[2])
    >>> gx2_expected = jnp.ones_like(x2) * (A.shape[0] * A.shape[2])
    >>> gx3_expected = jnp.ones_like(x3) * (A.shape[0] * A.shape[1])
    >>> jnp.array_equal(g.x1, gx1_expected).item()
    True
    >>> jnp.array_equal(g.x2, gx2_expected).item()
    True
    >>> jnp.array_equal(g.x3, gx3_expected).item()
    True
    """

    shape: tuple[int, ...] = eqx.field(static=True)
    f: IndexCallableT = eqx.field(static=True)
    dtype: DTypeLike | None = eqx.field(static=True)

    f_method_name: ClassVar[str] = "function"

    def __init__(
        self,
        shape: tuple[int, ...],
        f: IndexCallableT | None = None,
        *,
        dtype: DTypeLike | None = None,
    ) -> None:
        self.shape = (
            (int(shape),) if jnp.isscalar(shape) else tuple(int(s) for s in shape)
        )
        if hasattr(self, self.f_method_name) and f is not None:
            cn = self.__class__.__name__
            errmsg = (
                f"it appears that this '{cn}' already defines '{self.f_method_name}', "
                "so 'f' argument should not be provided"
            )
            raise ValueError(errmsg)
        if f is not None:
            f = jax.jit(f)
        self.f = f
        self.dtype = jnp.dtype(dtype) if dtype is not None else None

    def __check_init__(self) -> None:
        if getattr(self, "_f", None) is None and not hasattr(self, self.f_method_name):
            errmsg = (
                f"'{self.f_method_name}' must be provided either as an argument `f` "
                "or implemented as a method on a subclass"
            )
            raise ValueError(errmsg)

    def __repr__(self) -> str:
        cn = self.__class__.__name__
        attrs = f"shape={self.shape}"
        if self.dtype is not None:
            attrs += f" dtype={self.dtype}"
        fname = f" ({self._f.__name__})" if self._f is not None else ""
        return f"<{cn}{fname} {attrs}>"

    def __getitem__(self, args: Any) -> jnp.ndarray:
        index = DynamicIndexExpression(self.shape)[args]
        coords = tuple(c for c in index.coords if c is not None)
        shape = index.shape
        out = self._f(*coords).reshape(shape)
        if self.dtype is not None and out.dtype != self.dtype:
            out = out.astype(self.dtype)
        return out

    @property
    def _f(self) -> IndexCallableT:
        """The callable that takes a Numpy-style index and returns the corresponding
        subarray.
        """
        f = self.f if self.f is not None else getattr(self, self.f_method_name)
        return f

    def astype(self, dtype: DTypeLike) -> Self:
        """Return a new instance with the given dtype."""
        return replace(self, dtype=dtype)

    def _equals(self, other: object) -> bool:
        return (
            super()._equals(other) and self.f is other.f and self.dtype == other.dtype
        )


class LazyView(LazyArray):
    """Lazy view of an array, with optional element-wise transformation.

    Attributes
    ----------
    x
        The input array.
    op
        An optional element-wise operation to apply to the input array.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from grgg.utils.lazy import LazyView
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

    def function(self, i: Integers | None = None) -> jnp.ndarray:
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
    >>> from grgg.utils.lazy import LazyOuter
    >>> x1 = jnp.array([1, 2, 3])
    >>> x2 = jnp.array([4, 5])
    >>> O = LazyOuter(x1, x2)
    >>> O
    <LazyOuter (multiply) shape=(3, 2), dtype=...>
    >>> jnp.array_equal(O[...], jnp.outer(x1, x2)).item()
    True

    Scalar case.
    >>> O = LazyOuter(3, 4)
    >>> O
    <LazyOuter (multiply) shape=(), dtype=...>
    >>> O[...].item()
    12

    The convenion is that in this case any indexing
    is allowed, but always produces the same scalar result.
    >>> O[:5, ..., [1,2]].item()
    12

    JIT-compilation and differentiation work as expected.
    However, it is best to use :func:`equinox.filter_jit`
    and :func:`equinox.filter_grad` for functions that take
    :class:`LazyOuter` as input.
    >>> import equinox as eqx
    >>> x = jnp.arange(10, dtype=jnp.float32)
    >>> y = jnp.arange(8, dtype=jnp.float32)
    >>> outer = LazyOuter(x, y, op=jnp.add)
    >>> @eqx.filter_jit
    ... def sum_outer(outer, *args):
    ...     return outer[args].sum()
    >>> sum_outer(outer, ...).item()
    640.0
    >>> sum_outer(outer, 2, 3).item()
    5.0
    >>>
    >>> grad = eqx.filter_grad(sum_outer)
    >>> g = grad(outer, ...)
    >>> jnp.array_equal(g.x, jnp.ones_like(x) * outer.shape[1]).item()
    True
    >>> jnp.array_equal(g.y, jnp.ones_like(y) * outer.shape[0]).item()
    True
    """

    x: RealVector
    y: RealVector
    op: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = eqx.field(static=True)

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
        opname = f" ({self.op.__name__})" if self.op is not None else ""
        return f"<{cn}{opname} shape={self.shape}, dtype={self.dtype}>"

    def __getitem__(self, args: Any) -> jnp.ndarray:
        if self.is_scalar:
            args = Ellipsis
        return super().__getitem__(args)

    @property
    def is_scalar(self) -> bool:
        """Whether the outer product is a scalar."""
        return jnp.isscalar(self.x) and jnp.isscalar(self.y)

    def function(
        self, i: Integers | None = None, j: Integers | None = None
    ) -> jnp.ndarray:
        x = self.x if i is None else self.x[i]
        y = self.y if j is None else self.y[j]
        return self.op(x, y)
