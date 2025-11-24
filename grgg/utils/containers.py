from collections.abc import Sequence
from functools import singledispatchmethod
from typing import Any, Self, TypeVar

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import ArrayLike, DTypeLike

from grgg.abc import AbstractModule
from grgg.utils.misc import format_array

__all__ = ("ArrayBundle",)

A = TypeVar("A", bound=ArrayLike)


class ArrayBundle[A](AbstractModule, Sequence[A]):
    """Bundle of named arrays accessible by index or name."""

    names: eqx.AbstractClassVar[tuple[str, ...]]

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        names = list(getattr(cls, "names", []))
        names.extend(name for name in cls.__annotations__ if name != "names")
        cls.names = tuple(names)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.__repr_inner__()})"

    def __repr_inner__(self) -> str:
        arrays = [f"{name}={format_array(getattr(self, name))}" for name in self.names]
        return ", ".join(arrays)

    def __len__(self) -> int:
        return len(self.names)

    @singledispatchmethod
    def __getitem__(self, index: Any) -> A:
        errmsg = f"index must be 'int' or 'str', got '{type(index).__name__}'"
        raise TypeError(errmsg)

    @__getitem__.register
    def _(self, index: int) -> A:
        return getattr(self, self.names[index])

    @__getitem__.register
    def _(self, name: str) -> A:
        if name not in self.names:
            errmsg = f"unknown array name '{name}'"
            raise KeyError(errmsg)
        return getattr(self, name)

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
        return super()._equals(other) and all(
            jnp.array_equal(getattr(self, name).data, getattr(other, name).data)
            for name in self.names
        )

    def astype(self, dtype: DTypeLike) -> Self:
        """Return `self` with arrays cast to a given data type."""
        return self.__class__(*(array.astype(dtype) for array in self))

    def to_dict(self) -> dict[str, A]:
        """Return `self` as a dictionary of arrays."""
        return {name: getattr(self, name) for name in self.names}

    def to_common_dtype(self) -> Self:
        """Return `self` with arrays cast to a common data type."""
        return self.astype(self.dtype)
