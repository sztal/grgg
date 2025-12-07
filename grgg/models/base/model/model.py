from abc import abstractmethod
from collections.abc import Container
from typing import Any, Self, TypeVar

import equinox as eqx

from grgg.utils.dispatch import dispatch
from grgg.utils.misc import partition_choices
from grgg.utils.variables import ArrayBundle

from .fitting import AbstractModelFit, LeastSquaresFit
from .functions import AbstractModelFunctions
from .modules import AbstractModelModule
from .parameters import AbstractParameter, Parameters

__all__ = ("AbstractModel",)

O = TypeVar("O", bound="ArrayBundle")


class AbstractModel(AbstractModelModule[Self]):
    """Abstract base class for models."""

    Parameters: eqx.AbstractClassVar[type[Parameters]]
    functions: eqx.AbstractClassVar[type[AbstractModelFunctions]]

    n_units: eqx.AbstractVar[int]
    parameters: eqx.AbstractVar["Self.Parameters"]

    @abstractmethod
    def __init__(self, parameters: Parameters | None = None, **kwargs: Any) -> None:
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
        for name, parameter in self.parameters.mapping.items():
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
    def get_target_cls(self, method: str) -> type[ArrayBundle]:
        """Get the target statistics class for a given fitting method."""

    def prepare_fit_target(
        self,
        method: str,
        data: Any,
        *,
        homogeneous: bool | str | Container[str] | None = None,
        heterogeneous: bool | str | Container[str] | None = None,
    ) -> ArrayBundle:
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
        if homogeneous is None and heterogeneous is None:
            _homogeneous = []
            for name, param in self.parameters.mapping.items():
                if param.is_homogeneous:
                    _homogeneous.append(name)
        else:
            if homogeneous is False and heterogeneous is None:
                heterogeneous = True
            elif heterogeneous is False and homogeneous is None:
                homogeneous = True
            _homogeneous, _ = partition_choices(
                self.parameters.names,
                homogeneous=homogeneous,
                heterogeneous=heterogeneous,
            )
        observables = {}
        for name, param in self.parameters.mapping.items():
            stat, statmethod = param.get_statistic(
                self, method, homogeneous=name in _homogeneous
            )
            observable = statmethod.observed(data)
            observables[stat] = observable
        target_cls = self.get_target_cls(method)
        return target_cls(**observables)

    @dispatch
    def fit(self, target: ArrayBundle) -> LeastSquaresFit:
        """Prepare a model fitting procedure."""
        return LeastSquaresFit(self, target)

    @fit.dispatch
    def _(
        self,
        data: Any,
        *args: Any,
        method: str | None = None,
        init_params: bool = True,
        **kwargs: Any,
    ) -> AbstractModelFit:
        """Prepare a model fitting procedure."""
        if not method:
            method = self.get_default_fit_method(data)
        target = self.prepare_fit_target(method, data, *args, **kwargs)
        if init_params:
            self = self.init_params(target)
        return self.fit(target)

    def init_params(self, target: ArrayBundle) -> Self:
        """Initialize model parameters for fitting."""
        params = {}
        for param, stat in zip(
            self.model.parameters.mapping.items(),
            target,
            strict=True,
        ):
            name, param = param
            params[name] = self.init_param(param, stat)
        return self.replace(parameters=self.parameters.replace(**params))

    def init_param(
        self,
        param: AbstractParameter,
        target: ArrayBundle,
    ) -> AbstractParameter:
        """Abstract method to initialize model parameters for fitting."""
        return self._init_param(param, target, target.shape == ())

    @dispatch.abstract
    def _init_param(
        self,
        param: AbstractParameter,
        target: ArrayBundle,
        homogeneous: bool,
    ) -> AbstractParameter:
        """Abstract method to initialize model parameters for fitting."""

    # Model sampling interface -------------------------------------------------------

    @abstractmethod
    def sample(self, *args: Any, **kwargs: Any) -> Any:
        """Sample from the model."""
