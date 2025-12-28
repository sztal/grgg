from abc import abstractmethod
from typing import Any, Literal, Self, TypeVar

import equinox as eqx
import jax.numpy as jnp

from grgg._typing import ArrayLike
from grgg.utils.dispatch import dispatch
from grgg.utils.random import RandomGenerator
from grgg.utils.validation import validate
from grgg.utils.variables import ArrayBundle

from .fitting import AbstractModelFit, LeastSquaresFit
from .functions import AbstractModelFunctions
from .modules import AbstractModelModule
from .parameters import AbstractObservables, AbstractParameters

__all__ = ("AbstractModel",)

O = TypeVar("O", bound="ArrayBundle")


class AbstractModel(AbstractModelModule[Self]):
    """Abstract base class for models."""

    Parameters: eqx.AbstractClassVar[type[AbstractParameters]]
    Observables: eqx.AbstractClassVar[type[AbstractObservables]]
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

    @dispatch
    def get_fit_cls(self, method: Literal["least_squares"]) -> type[LeastSquaresFit]:  # noqa
        """Get the default fitting class for a given model and fitting method."""
        return LeastSquaresFit

    @validate
    def fit(
        self,
        obj: Any,
        default: Literal["homogeneous", "heterogeneous"] = "homogeneous",
        *,
        every: bool = True,
        method: str | None = None,
        rng: RandomGenerator | None = None,
        **initializers: str | ArrayLike,
    ) -> AbstractModelFit:
        """Instantiate model fitting interface.

        Parameters
        ----------
        obj
            Object from which to derive target statistics.
        method
            Fitting method to use. If `None`, uses the model's default fitting method.
        *args, **kwargs
            Additional arguments to passed to :meth:`AbstractModelFit.init` method.
            No initialization is done if both `*args` and `**kwargs` are empty.
        """
        if method is None:
            method = self.get_default_fit_method(obj)
        if isinstance(obj, self.Observables):
            target = obj
        else:
            # Handle extraction of target statistics from object
            target = {}
            for param in self.parameters:
                stat, statmethod = param.get_statistic(self, method, homogeneous=False)
                if stat in target:
                    continue
                statistic = statmethod.observed(obj)
                target[stat] = statistic
            target = self.Observables(**target)
        # Handle parameter initialization
        params = {}
        for name, param in self.parameters.mapping.items():
            if not every and name not in initializers:
                continue
            initializer = initializers.get(name, default)
            if not isinstance(initializer, str):
                value = jnp.asarray(initializer).astype(float)
            else:
                try:
                    value = param.initialize(self, target, initializer, rng=rng)
                except TypeError as exc:
                    if "argument 'rng'" in str(exc):
                        value = param.initialize(self, target, initializer)
            params[name] = value
        params = self.parameters.replace(**params)
        model = self.replace(parameters=params)
        fit_cls = self.get_fit_cls(method)
        fit = fit_cls(model, target)
        return fit

    # Model sampling interface -------------------------------------------------------

    @abstractmethod
    def sample(self, *args: Any, **kwargs: Any) -> Any:
        """Sample from the model."""
