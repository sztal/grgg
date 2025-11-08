from abc import abstractmethod
from typing import Any, Self

import equinox as eqx

from .functions import AbstractModelFunctions
from .modules import AbstractModelModule
from .parameters import AbstractParameters

__all__ = ("AbstractModel",)


class AbstractModel(AbstractModelModule[Self]):
    """Abstract base class for models."""

    Parameters: eqx.AbstractClassVar[type[AbstractParameters]]
    functions: eqx.AbstractClassVar[type[AbstractModelFunctions]]

    n_units: eqx.AbstractVar[int]
    parameters: eqx.AbstractVar["Self.Parameters"]

    def __check_init__(self) -> None:
        if self.n_units <= 0:
            errmsg = f"'n_units' must be positive, got {self.n_units}."
            raise ValueError(errmsg)
        if not isinstance(self.parameters, self.Parameters):
            errmsg = (
                f"'parameters' must be an instance of {self.Parameters.__name__}, "
                f"got {type(self.parameters).__name__} instead."
            )
            raise TypeError(errmsg)
        if not self.parameters:
            errmsg = "model must have at least one parameter."
            raise ValueError(errmsg)
        for name in self.Parameters.names:
            parameter = getattr(self.parameters, name)
            if not parameter.is_scalar and len(parameter) != self.n_units:
                errmsg = (
                    f"all non-scalar parameters must have leading axis size equal to "
                    f"'n_units' ({self.n_units}), but parameter '{name}' has "
                    f"{parameter.size} instead."
                )
                raise ValueError(errmsg)

    def __repr__(self) -> str:
        cn = self.__class__.__name__
        return f"{cn}({self.__repr_desc__()}, {self.parameters.__repr_inner__()})"

    def __repr_desc__(self) -> str:
        desc = f"{self.n_units}"
        return desc

    def __getattr__(self, name: str) -> Any:
        if name in self.Parameters.names:
            return getattr(self.parameters, name)
        return object.__getattribute__(self, name)

    @property
    def model(self) -> Self:
        """Self as model."""
        return self

    @property
    def params(self) -> "Self.Parameters":
        """Model parameters."""
        return self.parameters

    @property
    def is_heterogeneous(self) -> bool:
        """Whether the model has heterogeneous parameters."""
        return self.parameters.are_heterogeneous

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

    @classmethod
    def _make_parameters(
        cls, parameters: AbstractParameters | None, **kwargs: Any
    ) -> AbstractParameters:
        if parameters is None:
            return cls.Parameters(**kwargs)
        if kwargs:
            errmsg = "cannot specify both 'parameters' and parameter keyword arguments"
            raise ValueError(errmsg)
        return parameters
