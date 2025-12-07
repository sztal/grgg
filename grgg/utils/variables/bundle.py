from collections.abc import Sequence
from typing import Any, Self, TypeVar

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import ArrayLike as _ArrayLike
from jaxtyping import DTypeLike

from grgg.abc import AbstractModule
from grgg.utils.dispatch import dispatch
from grgg.utils.misc import format_array

from .variable import Variable

__all__ = ("ArrayBundle", "AbstractArrayBundle")


ArrayLike = _ArrayLike | Variable

A = TypeVar("A", bound=ArrayLike)


class AbstractArrayBundle(AbstractModule, Sequence[A]):
    """Abstract base class for bundles of named arrays."""

    mapping: eqx.AbstractVar[dict[str, A]]
    names: eqx.AbstractVar[Sequence[str]]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.__repr_inner__()})"

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | set(self.names))

    def __repr_inner__(self) -> str:
        inner = []
        for name, arr in self.mapping.items():
            if isinstance(arr, Variable):
                arr = arr.data
            inner.append(f"{name}={format_array(arr)}")
        return ", ".join(inner)

    def __len__(self) -> int:
        return len(self.mapping)

    @dispatch
    def __getitem__(self, index: int) -> ArrayLike:
        return self.mapping[self.names[index]]

    @__getitem__.dispatch
    def _(self, name: str) -> ArrayLike:
        return self.mapping[name]

    def __getattr__(self, name: str) -> ArrayLike:
        try:
            return self[name]
        except KeyError:
            return object.__getattribute__(self, name)

    @property
    def dtype(self) -> DTypeLike:
        """Resolved parameter data type."""
        if not self:
            errmsg = "cannot determine dtype of empty parameters' set"
            raise ValueError(errmsg)
        dtype = self[0].dtype
        for i in range(1, len(self)):
            param = self[i]
            dtype = jnp.promote_types(dtype, param.dtype)
        return dtype

    def _equals(self, other: object) -> bool:
        return (
            type(self) is type(other)
            and len(self.mapping) == len(other.mapping)
            and all(
                jnp.array_equal(self.mapping[n], other.mapping[n]) for n in self.names
            )
        )

    def astype(self, dtype: DTypeLike) -> Self:
        """Return `self` with arrays cast to a given data type."""
        return self.__class__(**{n: v.astype(dtype) for n, v in self.mapping.items()})

    def to_common_dtype(self) -> Self:
        """Return `self` with arrays cast to a common data type."""
        return self.astype(self.dtype)


class ArrayBundle[A](AbstractArrayBundle[A]):
    """Bundle of named arrays accessible by index or name.

    Examples
    --------
    >>> from dataclasses import dataclass
    >>> from grgg.utils.variables import ArrayBundle
    >>> import jax.numpy as jnp
    >>> bundle = ArrayBundle(x=1, y=[1,2])
    >>> bundle.replace(z=[1,2,3])
    ArrayBundle(x=1, y=i...[2], z=i...[3])
    >>> bundle["x"]
    Array(1, ...)
    >>> bundle.x
    Array(1, ...)
    >>> bundle[1]
    Array([1, 2], ...)
    >>> bundle.names
    ('x', 'y')
    >>> bundle.dtype
    dtype('int...')
    >>> bundle.astype(float)
    ArrayBundle(x=1.0, y=f...[2])
    >>> bundle[:2].equals(bundle)
    True

    Array bundles are compatible with JAX transformations.
    >>> import jax
    >>> def sum_bundle(b):
    ...     total = 0
    ...     for arr in b:
    ...         total += jnp.sum(arr)
    ...     return total
    >>> jax.jit(sum_bundle)(bundle).item()
    4
    >>> jax.grad(sum_bundle)(bundle.astype(float))
    ArrayBundle(x=1.00, y=f...[2])
    """

    arrays: tuple[A, ...] = ()
    names: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        arrays: tuple[A, ...] = (),
        names: tuple[str, ...] = (),
        **kwargs: Any,
    ) -> None:
        mapping = {**dict(zip(names, arrays, strict=True)), **kwargs}
        self.arrays = tuple(
            a if hasattr(a, "__jax_array__") else jnp.asarray(a)
            for a in mapping.values()
        )
        self.names = tuple(mapping.keys())

    @AbstractArrayBundle.__getitem__.dispatch
    def _(self, index: slice) -> "ArrayBundle":
        return self.__class__(
            arrays=self.arrays[index],
            names=self.names[index],
        )

    @property
    def mapping(self) -> dict[str, A]:
        """Mapping of names to arrays."""
        return dict(zip(self.names, self.arrays, strict=True))
