import inspect
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from grgg._typing import Integer

CarryT = jnp.ndarray | None

ForiBodyT = Callable[[Integer, jnp.ndarray], jnp.ndarray]
ForiCallT = Callable[[ForiBodyT], jnp.ndarray]

ForEachBodyT = Callable[[CarryT, CarryT], tuple[CarryT, CarryT]]
ForEachCallT = Callable[[ForEachBodyT], CarryT]

MapT = Callable[[jnp.ndarray, ...], jnp.ndarray]
ReduceT = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]

ScanLoopBodyT = Callable[[jnp.ndarray, ...], tuple[jnp.ndarray, jnp.ndarray | None]]
ScanLoopT = Callable[[], tuple[jnp.ndarray, jnp.ndarray | None]]

__all__ = ("fori", "foreach")


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
