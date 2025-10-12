import inspect
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from grgg._typing import Integer, Integers
from grgg.utils.random import RandomGenerator

CarryT = jnp.ndarray | None

ForiBodyT = Callable[[Integer, jnp.ndarray], jnp.ndarray]
ForiCallT = Callable[[ForiBodyT], jnp.ndarray]

ForEachBodyT = Callable[[CarryT, CarryT], tuple[CarryT, CarryT]]
ForEachCallT = Callable[[ForEachBodyT], CarryT]

MapT = Callable[[jnp.ndarray, ...], jnp.ndarray]
ReduceT = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]

ScanLoopBodyT = Callable[[jnp.ndarray, ...], tuple[jnp.ndarray, jnp.ndarray | None]]
ScanLoopT = Callable[[], tuple[jnp.ndarray, jnp.ndarray | None]]

__all__ = ("fori", "foreach", "mapreduce", "sample")


def fori(
    lower: int,
    upper: int,
    init: jnp.ndarray | None = None,
    **kwargs: Any,
) -> ForiCallT:
    """Fori loop abstraction.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from grgg.utils.compute import fori
    >>> total = jax.new_ref(0)
    >>> @fori(0, 5)
    ... def body(i):
    ...     total[...] += i
    >>> total
    Ref(10, ...)

    Explicit reductions without refs can be implemented too.
    >>> @fori(0, 5, init=jnp.array(0))
    ... def body(i, val):
    ...     return val + i
    >>> body
    Array(10, ...)
    """

    def __call(body: ForiBodyT) -> jnp.ndarray:
        if len(inspect.signature(body).parameters) == 1:

            def __body(i: Integer, _) -> jnp.ndarray | None:  # type: ignore[override]
                return body(i)
        else:

            def __body(i: Integer, val: jnp.ndarray | None) -> jnp.ndarray | None:
                return body(i, val)

        return jax.lax.fori_loop(lower, upper, __body, init, **kwargs)

    return __call


def foreach(
    xs: jnp.ndarray,
    init: jnp.ndarray | None = None,
    **kwargs: Any,
) -> ForEachCallT:
    """Foreach loop abstraction based on :func:`jax.lax.scan`.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from grgg.utils.compute import foreach
    >>> total = jax.new_ref(0)
    >>> @foreach(jnp.array([1, 2, 3, 4, 5]))
    ... def body(x):
    ...     total[...] += x
    >>> total
    Ref(15, ...)

    Explicit reductions without refs can be implemented too.
    >>> @foreach(jnp.array([1, 2, 3, 4, 5]), init=0)
    ... def body(carry, x):
    ...     return carry + x, None
    >>> body
    (Array(15, ...), None)

    Iteration over arbitrary pytrees is supported too.
    >>> xs = (jnp.arange(5), jnp.arange(5, 10))
    >>> @foreach(xs, init=0)
    ... def body(carry, x):
    ...     mul = x[0] * x[1]
    ...     return carry + mul, mul
    >>> body
    (Array(80, ...), Array([ 0,  6, 14, 24, 36], ...))
    """

    def __call(body: ForEachBodyT) -> tuple[CarryT, CarryT]:
        one_arg_body = len(inspect.signature(body).parameters) == 1
        if one_arg_body:

            def __body(_, xs: jnp.ndarray) -> tuple[None, jnp.ndarray | None]:  # type: ignore
                return None, body(xs)
        else:

            def __body(
                carry: CarryT,
                xs: jnp.ndarray,
            ) -> tuple[CarryT, jnp.ndarray | None]:
                return body(carry, xs)

        result = jax.lax.scan(__body, init, xs, **kwargs)
        return result[1] if one_arg_body else result

    return __call


def mapreduce(
    f: MapT | None = None,
    /,
    init: jnp.ndarray | None = None,
    reduction: ReduceT = jnp.add,
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
        return lambda func: mapreduce(func, *args, **kwargs)

    init = jnp.array(0.0) if init is None else jnp.asarray(init)

    scan_func = (
        _define_unweighted_scan_func(reduction, f)
        if n_samples <= 0
        else _define_weighted_scan_func(reduction, f)
    )

    rng = RandomGenerator.from_seed(rng)
    _p = p

    def __scan_loop(
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

    return __scan_loop


def _define_unweighted_scan_func(reduction: ReduceT, f: MapT) -> ScanLoopBodyT:
    def _scan_func(carry: jnp.ndarray, x: jnp.ndarray) -> tuple[jnp.ndarray, None]:
        new_carry = reduction(carry, f(x))
        return new_carry, None

    return _scan_func


def _define_weighted_scan_func(reduction: ReduceT, f: MapT) -> ScanLoopBodyT:
    def _scan_func(carry: jnp.ndarray, x: jnp.ndarray) -> tuple[jnp.ndarray, None]:
        x, w = x
        new_carry = reduction(carry, w * f(x))
        return new_carry, None

    return _scan_func


def sample(
    xs: jnp.ndarray,
    n_samples: int,
    *,
    p: jnp.ndarray | None = None,
    importance: bool = True,
    deterministic: bool = False,
    replace: bool = False,
    rng: RandomGenerator | Integers | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute importance sampling weights.

    Parameters
    ----------
    n_samples
        Number of importance samples to draw.
    p
        Sampling probabilities for each item.
    importance
        Whether to use importance sampling.
    deterministic
        Whether to use deterministic sampling.
        Used only when `importance` is `True` and `p` is given.
        An error is raised when used with `replace=True`.
    replace
        Whether to sample with replacement.
    rng
        Random number generator or seed.

    Returns
    -------
    samples
        Sampled items.
    weights
        Sampling weights for each sampled item.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from grgg.utils.compute import sample
    >>> from grgg.utils.random import RandomGenerator
    >>> rng = RandomGenerator(123)

    First we try to use uniform sampling to reconstruct the sum over a set of numbers.
    >>> X = jnp.arange(1000)
    >>> X.sum().item()
    499500
    >>> x, w = sample(X, 20, rng=rng)
    >>> (x * w).sum().item()
    451900.0

    Not too accurate. We can try to do better using importance sampling,
    where sampling weights should be at least roughly proportional
    to the output we want to estimate. For the sake of example, here we will
    use weights proportional to the values the contributions to the mean.
    which is not realistic in practice, but demomnstrates the method, as in this
    special case we will recover the sum exactly.
    >>> p = X
    >>> x, w = sample(X, 20, p=p, rng=rng)
    >>> (x * w).sum().item()
    499500.0

    Importance sampling can be done deterministically too.
    >>> x, w = sample(X, 20, p=p, rng=rng, deterministic=True)
    >>> (x * w).sum().item()
    499500.0

    Note that it cannot be done with replacement.
    >>> x, w = sample(X, 20, p=p, rng=rng, deterministic=True, replace=True)
    Traceback (most recent call last):
        ...
    ValueError: Deterministic importance sampling with replacement is not allowed.
    """
    n = len(xs)
    rng = RandomGenerator.from_seed(rng)
    if p is not None:
        p = jax.lax.stop_gradient(p / p.sum())
    if p is not None and importance:
        if deterministic:
            if replace:
                errmsg = (
                    "Deterministic importance sampling with replacement is not allowed."
                )
                raise ValueError(errmsg)
            indices = jnp.argsort(p)[-n_samples:]
        else:
            indices = rng.choice(len(xs), (n_samples,), p=p, replace=replace)
        w = 1 / (p[indices] * n_samples)
        x = jnp.take(xs, indices, axis=0)
        return x, w
    x = rng.choice(xs, (n_samples,), p=p, replace=replace)
    w = jnp.full((n_samples,), n / n_samples)
    return x, w
