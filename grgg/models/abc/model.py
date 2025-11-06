from abc import abstractmethod
from typing import Any, NamedTuple, Self

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import DTypeLike

from .functions import AbstractModelFunctions
from .modules import AbstractModelModule

__all__ = ("AbstractModel",)


class AbstractModel(AbstractModelModule[Self]):
    """Abstract base class for models."""

    Parameters: eqx.AbstractClassVar[type[NamedTuple]]
    functions: eqx.AbstractClassVar[type[AbstractModelFunctions]]

    n_units: eqx.AbstractVar[int]
    parameters: eqx.AbstractVar["Self.Parameters"]

    def __check_init__(self) -> None:
        if self.n_units <= 0:
            errmsg = f"'n_units' must be positive, got {self.n_units}."
            raise ValueError(errmsg)
        if not self.parameters:
            errmsg = "model must have at least one parameter."
            raise ValueError(errmsg)
        for name in self.Parameters._fields:
            parameter = getattr(self, name)
            if not parameter.is_scalar and len(parameter) != self.n_units:
                errmsg = (
                    f"all non-scalar parameters must have leading axis size equal to "
                    f"'n_units' ({self.n_units}), but parameter '{name}' has "
                    f"{parameter.size} instead."
                )
                raise ValueError(errmsg)

    def __repr__(self) -> str:
        params = [getattr(self, name) for name in self.Parameters._fields]
        inner = ", ".join([f"{self.n_units}", ", ".join(map(repr, params))])
        return f"{self.__class__.__name__}({inner})"

    @property
    def model(self) -> Self:
        """Self as model."""
        return self

    @property
    def dtype(self) -> DTypeLike:
        """Model parameter data type."""
        if not self.parameters:
            errmsg = "cannot determine dtype of empty parameters' set"
            raise ValueError(errmsg)
        dtype = self.parameters[0].dtype
        for param in self.parameters[1:]:
            dtype = jnp.promote_types(dtype, param.dtype)
        return dtype

    @property
    def is_heterogeneous(self) -> bool:
        """Whether the model has heterogeneous parameters."""
        return any(p.is_heterogeneous for p in self.parameters)

    @property
    def is_homogeneous(self) -> bool:
        """Whether the model has homogeneous parameters."""
        return not self.is_heterogeneous

    @abstractmethod
    def sample(self, *args: Any, **kwargs: Any) -> Any:
        """Sample from the model."""

    @abstractmethod
    def _equals(self, other: object) -> bool:
        return (
            super()._equals(other)
            and self.n_units == other.n_units
            and len(self.parameters) == len(other.parameters)
            and all(
                p1.equals(p2)
                for p1, p2 in zip(self.parameters, other.parameters, strict=True)
            )
        )
