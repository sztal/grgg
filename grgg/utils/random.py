import secrets
from collections.abc import Callable
from functools import partial
from typing import Any, Self

import equinox as eqx
import jax
import jax.numpy as jnp

from grgg._typing import Integers
from grgg.abc import AbstractModule

__all__ = ("RandomGenerator",)


class RandomGenerator(AbstractModule):
    """Random generator compatible with :mod:`jax`.

    However, it can cause problems in complex jit-compiled and multithreaded
    environments due to its mutable state. In such cases, direct handling of
    JAX random keys is recommended. However, :class:`RandomGenerator` may still
    be helpful for initializing keys from integers or `None`, especially using
    its class method :meth:`make_key`.

    Attributes
    ----------
    key
        JAX random key.

    Examples
    -------
    The random generator remains stateful even in jitted functions.
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from grgg.utils.random import RandomGenerator
    >>> @jax.jit
    ... def func(rng): return rng.normal((300,))
    >>> rng1 = RandomGenerator(0)
    >>> rng2 = RandomGenerator(0)
    >>> bool(jnp.all(func(rng1) == func(rng2)))
    True
    >>> x1, x2 = func(rng1), func(rng1)
    >>> bool(jnp.all(x1 == x2))
    False

    Create from a seed without making a copy if the seed is already a PRNG.
    >>> rng1 = RandomGenerator.from_seed(42)
    >>> rng2 = RandomGenerator.from_seed(rng1)
    >>> rng1 is rng2
    True

    But copying is supported too.
    >>> rng2 = rng1.copy()
    >>> rng1.equals(rng2)
    True

    Generator copies always have independent states.
    >>> _ = rng2.normal()
    >>> rng2.equals(rng1)
    False

    Split into two independent generators.
    >>> rng1, rng2 = rng1.split()
    >>> bool(jnp.all(func(rng1) == func(rng2)))
    False
    """

    _key: jax.ArrayRef = eqx.field(init=False, repr=False)

    def __init__(self, key: jax.Array | int | None = None) -> None:
        self._key = jax.array_ref(self.make_key(key))

    def __getattr__(self, name: str) -> Callable[..., jax.Array]:
        method = getattr(jax.random, name)
        subkey, newkey = self.split_key()
        self._key[...] = newkey
        return partial(method, subkey)

    @property
    def key(self) -> jax.Array:
        return self._key[...]

    @property
    def child(self) -> Self:
        """Create a child random generator and mutate self."""
        new_key, sub_key = self.split_key()
        self._key[...] = new_key
        return self.replace(key=sub_key)

    @classmethod
    def is_prng_key(cls, key: object) -> bool:
        """Check if the input is a JAX PRNG key."""
        return isinstance(key, jnp.ndarray) and isinstance(
            key.dtype, jax._src.prng.KeyTy
        )

    @classmethod
    def make_key(cls, key: jax.Array | int | Self | None = None) -> jax.Array:
        """Preprocess the input key.

        If `key` is an instance of the same class, return its internal key.
        If `key` is `None`, generate a random seed based on the system entropy.
        Otherwise, convert `key` to a JAX array.
        """
        if isinstance(key, cls):
            key = key.key
        if key is None:
            key = secrets.randbelow(2**31)
        if not isinstance(key, jnp.ndarray):
            key = jnp.asarray(key, dtype=jnp.uint32)
        if (
            jnp.isscalar(key)
            and not isinstance(key, jax._src.prng.PRNGKeyArray)
            and not cls.is_prng_key(key)
        ):
            key = jax.random.key(key)
        return key

    @classmethod
    def from_seed(cls, seed: int | None = None) -> Self:
        """Create without making copy if seed is is already a PRNG."""
        if isinstance(seed, cls):
            return seed
        return cls(seed)

    def replace(self, **kwargs: Any) -> Self:
        kwargs = {"key": self.key, **kwargs}
        return super().replace(**kwargs)

    def split(self, num: int = 2) -> tuple[Self, ...]:
        """Split the random generator into two new generators."""
        keys = self.split_key(num=num)
        return tuple(self.__class__(key) for key in keys)

    def split_key(self, num: int = 2) -> Integers:
        """Split the internal key into two new keys."""
        return jax.random.split(self.key, num=num)

    def fold_key(self, data: jnp.ndarray) -> Integers:
        """Fold data into the internal key and update the internal state."""
        return jax.random.fold_in(self.key, data)

    def fold_in(self, data: jnp.ndarray) -> Self:
        """Fold data into the internal key and update the internal state."""
        new_key = self.fold_key(data)
        return self.__class__(new_key)

    def _equals(self, other: object) -> bool:
        return super()._equals(other) and jnp.array_equal(self.key, other.key)
