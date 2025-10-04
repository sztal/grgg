from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from grgg._typing import Integers
from grgg.utils.random import RandomGenerator

ScanLoopBodyT = Callable[[jnp.ndarray, ...], tuple[jnp.ndarray, jnp.ndarray | None]]
ScanLoopT = Callable[[], tuple[jnp.ndarray, jnp.ndarray | None]]
ReductionT = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
MapFuctionT = Callable[[jnp.ndarray, ...], jnp.ndarray]
AxesT = int | tuple[int, ...]

__all__ = ("MapReduce", "Map", "Reduce")


def Map(
    f: MapFuctionT | None = None,
    /,
    *,
    batch_size: int | None = None,
) -> Callable[[MapFuctionT], MapFuctionT | Callable[[MapFuctionT], MapFuctionT]]:
    """Map abstraction.

    Applied as a decorator to a function it create a :func:`jax.lax.map`
    vectorized version of the function. Additional keyword arguments are passed
    to :func:`jax.lax.map`.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from grgg.utils.compute import Map
    >>> @Map
    ... def loop(x):
    ...     return x**2
    >>> loop(jnp.array([1, 2, 3]))
    Array([1, 4, 9], ...)

    >>> @Map(batch_size=2)
    ... def loop(x):
    ...     return x**2
    >>> loop(jnp.array([1, 2, 3]))
    Array([1, 4, 9], ...)
    """
    if f is None:
        return lambda f: Map(f, batch_size=batch_size)

    @jax.jit
    def mapped(xs: jnp.ndarray) -> jnp.ndarray:
        return jax.lax.map(f, xs, batch_size=batch_size)

    return mapped


def Reduce(
    f: ReductionT | None = None,
    /,
    *,
    axes: AxesT = 0,
) -> Callable[[ScanLoopBodyT], ScanLoopT | Callable[[ScanLoopBodyT], ScanLoopT]]:
    """Reduce abstraction.

    Applied as a decorator to a function it create a reduction function
    over the first axis of an array. Additional keyword arguments are passed to
    :func:`jax.lax.reduce`.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from grgg.utils.compute import Reduce
    >>> @Reduce
    ... def summation(x, y):
    ...     return x + y
    >>> summation(jnp.array([1, 2, 3]), 0)
    Array(6, ...)
    """
    if f is None:
        return lambda func: Reduce(func, axes)

    if not isinstance(axes, tuple):
        axes = (axes,)
    _axes = axes

    @jax.jit
    def reduction(
        xs: jnp.ndarray, *args: Any, axes: AxesT | None = None, **kwargs: Any
    ) -> jnp.ndarray:
        if axes is None:
            axes = _axes
        return jax.lax.reduce(xs, *args, f, dimensions=axes, **kwargs)

    return reduction


def MapReduce(
    f: MapFuctionT | None = None,
    /,
    init: jnp.ndarray | None = None,
    reduction: ReductionT = jnp.add,
    *,
    reverse: bool = False,
    unroll: int = 1,
    n_samples: int = 0,
    replace: bool = False,
    p: jnp.ndarray | None = None,
    rng: RandomGenerator | Integers | None = None,
) -> Callable[[ScanLoopBodyT], ScanLoopT]:
    """Map and reduce abstraction based on :func:`jax.lax.scan`.

    Parameters
    ----------
    xs
        Array-like values to compute over.
    init
        Initial value for the reduction.
    reduction
        Reduction function to apply.
    reverse
        Whether to scan in reverse order.
    unroll
        Amount to unroll the scan loop.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from grgg.utils.compute import MapReduce
    >>> from grgg.utils.random import RandomGenerator
    >>> rng = RandomGenerator(17)

    By default, summation with 0 init is used as the reduction function.
    >>> @MapReduce(init=0)
    ... def loop(x):
    ...     return x
    >>> loop(jnp.array([1, 2, 3]))
    Array(6, ...)

    >>> @MapReduce(init=0)
    ... def loop(x):
    ...     return x ** 2
    >>> loop(jnp.array([1, 2, 3, 4, 5]))
    Array(55, ...)

    Map-reduce with uniform sampling.
    >>> X = jnp.arange(1000)
    >>> X.sum()
    Array(499500, ...)
    >>> @MapReduce(init=0, n_samples=10, rng=rng)
    ... def loop(x):
    ...     return x
    >>> loop(X)
    Array(520600., ...)

    Not bad but not perfect (but unbiased) estimate of the total.
    We can try to do better using importance sampling,
    where sampling weights should be at least roughly proportional
    to the total output after reduction.
    >>> p = X / X.sum()
    >>> @MapReduce(init=0, n_samples=10, p=p, rng=rng)
    ... def loop(x):
    ...     return x
    >>> loop(X)
    Array(499500., ...)

    In this simple case we recover the total exactly, since our sampling weights
    were exactly proportional to the output.
    """
    if not isinstance(f, Callable):
        args = (init,) if f is None else (f, init)
        kwargs = {
            "reduction": reduction,
            "n_samples": n_samples,
            "p": p,
            "rng": rng,
            "reverse": reverse,
            "unroll": unroll,
        }
        return lambda func: MapReduce(func, *args, **kwargs)

    init = jnp.array(0.0) if init is None else jnp.asarray(init)

    scan_func = (
        _define_unweighted_scan_func(reduction, f)
        if n_samples <= 0
        else _define_weighted_scan_func(reduction, f)
    )

    rng = RandomGenerator.from_seed(rng)
    _p = p

    def scan_loop(
        xs: jnp.ndarray,
        init: jnp.ndarray = init,
        **kwargs: Any,
    ) -> ScanLoopT:
        kwargs = {
            "reverse": reverse,
            "unroll": unroll,
            **kwargs,
        }
        q = None
        if n_samples > 0:
            if _p is None or jnp.isscalar(_p):
                n_total = xs.shape[0]
                xs = rng.choice(xs, (n_samples,), replace=replace)
                q = n_total / n_samples
            else:
                # Importance sampling
                p = jax.lax.stop_gradient(_p / _p.sum())
                indices = rng.choice(xs.shape[0], (n_samples,), p=p, replace=replace)
                xs = jnp.take(xs, indices, axis=0)
                q = 1 / (p[indices] * n_samples)
        if jnp.isscalar(q):
            q = jnp.broadcast_to(q, (xs.shape[0],))
        _xs = xs if q is None else (xs, q)
        reduced, _ = jax.lax.scan(scan_func, init, _xs, **kwargs)
        return reduced

    return scan_loop


def _define_unweighted_scan_func(
    reduction: ReductionT, f: MapFuctionT
) -> ScanLoopBodyT:
    def _scan_func(carry: jnp.ndarray, x: jnp.ndarray) -> tuple[jnp.ndarray, None]:
        new_carry = reduction(carry, f(x))
        return new_carry, None

    return _scan_func


def _define_weighted_scan_func(reduction: ReductionT, f: MapFuctionT) -> ScanLoopBodyT:
    def _scan_func(carry: jnp.ndarray, x: jnp.ndarray) -> tuple[jnp.ndarray, None]:
        x, w = x
        new_carry = reduction(carry, w * f(x))
        return new_carry, None

    return _scan_func
