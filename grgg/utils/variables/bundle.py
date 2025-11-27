from collections.abc import Sequence
from functools import singledispatchmethod
from typing import Any, Self, TypeVar

import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, DTypeLike

from grgg.abc import AbstractModule
from grgg.utils.misc import format_array

from .variable import Variable

__all__ = ("ArrayBundle",)


A = TypeVar("A", bound=ArrayLike)


class ArrayBundle[A](AbstractModule, Sequence[A]):
    """Bundle of named arrays accessible by index or name."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.__repr_inner__()})"

    def __repr_inner__(self) -> str:
        inner = []
        for name, arr in self.to_dict().items():
            if not isinstance(arr, Variable):
                inner.append(format_array(arr))
            else:
                inner.append(f"{name}={format_array(arr.data)}")
        return ", ".join(inner)

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
    def names(self) -> list[str]:
        """Names of the arrays in the bundle."""
        return self.get_names()

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

    @classmethod
    def get_names(cls) -> list[str]:
        """Names of the arrays in the bundle."""
        return cls.get_instance_fields()

    def _equals(self, other: object) -> bool:
        return super()._equals(other) and jax.tree_util.tree_equal(self, other)

    def astype(self, dtype: DTypeLike) -> Self:
        """Return `self` with arrays cast to a given data type."""
        return self.__class__(*(array.astype(dtype) for array in self))

    def to_dict(self) -> dict[str, A]:
        """Return `self` as a dictionary of arrays."""
        return {name: getattr(self, name) for name in self.names}

    def to_common_dtype(self) -> Self:
        """Return `self` with arrays cast to a common data type."""
        return self.astype(self.dtype)
