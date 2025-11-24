from abc import abstractmethod
from collections.abc import Mapping
from typing import Any, Self

import equinox as eqx
from plum import dispatch

from .fitting import AbstractFitTarget, AbstractModelFit
from .functions import AbstractModelFunctions
from .modules import AbstractModelModule
from .parameters import AbstractParameters

__all__ = ("AbstractModel",)


class AbstractModel(AbstractModelModule[Self]):
    """Abstract base class for models."""

    Parameters: eqx.AbstractClassVar[type[AbstractParameters]]
    functions: eqx.AbstractClassVar[type[AbstractModelFunctions]]
    fit_default_method: eqx.AbstractClassVar[str]
    fit_targets: eqx.AbstractClassVar[Mapping[str, type[AbstractFitTarget]]]

    n_units: eqx.AbstractVar[int]
    parameters: eqx.AbstractVar["Self.Parameters"]

    def __init_subclass__(cls) -> None:
        for name in getattr(cls, "fit_targets", {}):
            if name not in (registry := AbstractModelFit.__available_fitters__):
                errmsg = (
                    f"fit target '{name}' has no corresponding fitter registered in "
                    f"AbstractModelFit; should be one of {list(registry)}"
                )
                raise ValueError(errmsg)

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

    @dispatch
    def fit(  # type: ignore  # noqa
        self,
        *,
        method: str | None = None,
        **kwargs: Any,
    ) -> AbstractModelFit:
        """Get a fitting procedure for the model without target data."""
        method = method or self.fit_default_method
        target = self.fit_targets[method](**kwargs)
        fitter_cls = AbstractModelFit.__available_fitters__[method]
        return fitter_cls(self, target)

    @dispatch
    def fit(  # type: ignore  # noqa
        self,
        target: AbstractFitTarget,
        *,
        method: str | None = None,
    ) -> AbstractModelFit:
        """Get a fitting procedure for the model given a fit target."""
        method = method or self.fit_default_method
        fitter_cls = AbstractModelFit.__available_fitters__[method]
        return fitter_cls(self, target)

    @dispatch
    def fit(  # type: ignore  # noqa
        self,
        target: Any,
        *,
        method: str | None = None,
        **kwargs: Any,
    ) -> AbstractModelFit:
        """Get a fitting procedure for the model."""
        method = method or self.fit_default_method
        if method not in (methods := AbstractModelFit.__available_fitters__):
            errmsg = f"unknown fit method '{method}', should be one of {list(methods)}"
            raise ValueError(errmsg)
        fitter_cls = methods[method]
        target = fitter_cls.make_target(self, target, **kwargs)
        return fitter_cls(self, target)

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
