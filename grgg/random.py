import secrets
from collections.abc import Callable
from functools import partial
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp

__all__ = ("RandomGenerator",)


class RandomGenerator(eqx.Module):
    """Random generator compatible with :mod:`jax`.

    Attributes
    ----------
    key
        JAX random key.

    Examples
    -------
    The random generator remains stateful even in jitted functions.
    >>> @jax.jit
    ... def func(rng): return rng.normal((300,))
    >>> rng1 = RandomGenerator.from_seed(0)
    >>> rng2 = RandomGenerator.from_seed(0)
    >>> bool(jnp.all(func(rng1) == func(rng2)))
    True
    >>> x1, x2 = func(rng1), func(rng1)
    >>> bool(jnp.all(x1 == x2))
    False

    Construct from an existing generator without making copy.
    >>> rng = RandomGenerator.from_seed(rng1)
    >>> rng is rng1
    True

    Split into two independent generators.
    >>> rng1, rng2 = rng1.split()
    >>> bool(jnp.all(func(rng1) == func(rng2)))
    False
    """

    key: jax.Array

    def __init__(
        self,
        key: jax.Array | int | None = None,
    ) -> None:
        if isinstance(key, type(self)):
            errmsg = (
                "cannot initialize from another RandomGenerator instance; "
                "use `from_seed` constructor to handle this case"
            )
            raise ValueError(errmsg)
        if key is None:
            key = secrets.randbelow(2**31)
        if not isinstance(key, jax.Array) or not isinstance(
            key.dtype, jax._src.prng.KeyTy
        ):
            key = jax.random.key(jnp.asarray(key, dtype=jnp.uint32))
        self.key = jax.array_ref(key)

    def __getattr__(self, name: str) -> Callable[..., jax.Array]:
        newkey, subkey = jax.random.split(self.key[...])
        self.key[...] = newkey
        method = getattr(jax.random, name)
        return partial(method, subkey)

    def split(self) -> tuple[Self, Self]:
        """Split the random generator into two new generators."""
        key1, key2 = jax.random.split(self.key[...])
        return self.__class__(key1), self.__class__(key2)

    @classmethod
    def from_seed(cls, key: jax.Array | int | Self | None = None) -> Self:
        """Create from a seed

        Seed can be an integer scalar (also in the form of :class:`jax.Array`),
        or another :class:`RandomGenerator` instance. When `None`, a random seed
        cryptographically secure seed is generated based on the system entropy.
        """
        if isinstance(key, cls):
            return key
        return cls(key)
