from abc import abstractmethod
from collections.abc import Container
from typing import Any, Self, TypeVar

import equinox as eqx
import jax

from grgg.models.base.observables import AbstractObservables
from grgg.utils.dispatch import dispatch
from grgg.utils.dotpath import dotget
from grgg.utils.misc import partition_choices

from .fitting import ModelFit
from .functions import AbstractModelFunctions
from .modules import AbstractModelModule
from .parameters import AbstractParameter, AbstractParameters

__all__ = ("AbstractModel",)

O = TypeVar("O", bound="AbstractObservables")


class AbstractModel(AbstractModelModule[Self]):
    """Abstract base class for models."""

    Parameters: eqx.AbstractClassVar[type[AbstractParameters]]
    functions: eqx.AbstractClassVar[type[AbstractModelFunctions]]

    n_units: eqx.AbstractVar[int]
    parameters: eqx.AbstractVar["Self.Parameters"]

    @abstractmethod
    def __init__(
        self, parameters: AbstractParameters | None = None, **kwargs: Any
    ) -> None:
        if parameters is not None and kwargs:
            errmsg = "cannot specify both 'parameters' and parameter keyword arguments"
            raise ValueError(errmsg)
        if parameters is None:
            parameters = self.Parameters(**kwargs)
        self.parameters = parameters

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
        for name, parameter in self.parameters.to_dict().items():
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
        if name in self.parameters.names:
            return self.parameters[name]
        return object.__getattribute__(self, name)

    def _equals(self, other: object) -> bool:
        return (
            super()._equals(other)
            and self.n_units == other.n_units
            and self.parameters.equals(other.parameters)
        )

    # Properties ---------------------------------------------------------------------

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

    # Model fitting interface --------------------------------------------------------

    @dispatch
    def get_default_fit_method(self, data: Any) -> str:  # noqa
        """Get the default fitting method for a given model and target statistics."""
        return "least_squares"

    @dispatch.abstract
    def get_target_cls(self, method: str) -> type[AbstractObservables]:
        """Get the target statistics class for a given fitting method."""

    @dispatch
    def prepare_fit_target(
        self,
        method: str,
        data: Any,
        **kwargs: Any,
    ) -> AbstractObservables:
        """Prepare target statistics for model fitting.

        Parameters
        ----------
        method
            The fitting method to use.
        data
            The data to fit the model to.
        **kwargs
            Additional arguments passed to :meth:`~prepare_fit_target`.
        """
        target_cls = self.get_target_cls(method)
        observables = {}
        for name, field in target_cls.get_fields().items():
            statmethod = dotget(self, field.metadata["statistic"])
            observable = statmethod.observed(data)
            observables[name] = observable
        target = target_cls(**observables)
        return self.prepare_fit_target(target, **kwargs)

    @prepare_fit_target.dispatch
    def _(
        self,
        target: AbstractObservables,
        *,
        homogeneous: bool | str | Container[str] | None = None,
        heterogeneous: bool | str | Container[str] | None = None,
    ) -> AbstractObservables:
        """Prepare target statistics for model fitting.

        Parameters
        ----------
        target
            Target statistics.
        homogeneous
            Observables to treat as homogeneous.
        heterogeneous
            Observables to treat as heterogeneous.
        """
        if homogeneous is None and heterogeneous is None or homogeneous is False:
            heterogeneous = True
        if heterogeneous is False and homogeneous is None:
            homogeneous = True
        _homogeneous, _ = partition_choices(
            target.names, homogeneous=homogeneous, heterogeneous=heterogeneous
        )
        stats = {
            name: target.reduce(stat) if name in _homogeneous else stat
            for name, stat in target.to_dict().items()
        }
        return target.replace(**stats)

    @dispatch
    def fit(self, target: AbstractObservables) -> ModelFit:
        """Prepare a model fitting procedure."""
        return ModelFit(self, target)

    @fit.dispatch
    def _(
        self,
        data: Any,
        *args: Any,
        method: str | None = None,
        init_params: bool = True,
        **kwargs: Any,
    ) -> ModelFit:
        """Prepare a model fitting procedure."""
        if not method:
            method = self.get_default_fit_method(data)
        target = self.prepare_fit_target(method, data, *args, **kwargs)
        if init_params:
            self = self.init_params(target)
        return self.fit(target)

    def init_params(self, target: AbstractObservables) -> Self:
        """Initialize model parameters for fitting."""
        params = jax.tree.map(
            lambda p: self.init_param(p, target),
            self.model.parameters,
            is_leaf=lambda x: isinstance(x, AbstractParameter),
        )
        return self.replace(parameters=params)

    @dispatch.abstract
    def init_param(
        param: AbstractParameter,
        target: AbstractObservables,
    ) -> AbstractParameter:
        """Abstract method to initialize model parameters for fitting."""

    # Model sampling interface -------------------------------------------------------

    @abstractmethod
    def sample(self, *args: Any, **kwargs: Any) -> Any:
        """Sample from the model."""
