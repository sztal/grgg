from abc import abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar, Self, TypeVar

import equinox as eqx
import jax
from jaxtyping import PyTree
from plum import dispatch

from grgg._typing import Numbers, Real
from grgg.utils.containers import ArrayBundle

from .modules import AbstractModelModule
from .parameters import AbstractParameter

if TYPE_CHECKING:
    from .model import AbstractModel

__all__ = ("AbstractModelFit", "AbstractFitTarget")


T = TypeVar("T", bound="AbstractModel")

TargetT = PyTree[Numbers]
FitOutputT = tuple["AbstractModelFit[T]", PyTree[Any]]
ObjectiveT = Callable[[T, ...], Real]


class AbstractFitTarget(ArrayBundle[Numbers]):
    """Abstract base class for fit targets used in model fitting."""


class AbstractModelFit[T](AbstractModelModule[T]):
    """Abstract base class for model fitting procedures."""

    model: T
    target: eqx.AbstractVar[TargetT]

    alias: eqx.AbstractClassVar[str]
    __available_fitters__: ClassVar[dict[str, type[Self]]] = {}

    def __init_subclass__(cls) -> None:
        if alias := getattr(cls, "alias", None):
            if (
                alias in cls.__available_fitters__
                and cls is not cls.__available_fitters__[alias]
            ):
                errmsg = (
                    f"Fitter alias '{alias}' is already registered for "
                    f"{cls.__available_fitters__[alias].__name__}."
                )
                raise ValueError(errmsg)
            cls.__available_fitters__[alias] = cls

    @abstractmethod
    def define_objective(cls, model: T, *args: Any, **kwargs: Any) -> ObjectiveT:
        """Define the objective function for fitting."""

    @classmethod
    def from_data(cls, data: Any, model: T, **kwargs: Any) -> Self:
        """Create a fitting instance from raw data."""
        target = cls.make_target(data, **kwargs)
        return cls(model=model, target=target)

    @classmethod
    def make_target(cls, model: T, data: Any, **kwargs: Any) -> TargetT:
        """Convert raw data into a target format suitable for fitting."""
        return cls._make_target(cls, model, data, **kwargs)

    @staticmethod
    @dispatch.abstract
    def _make_target(
        fitter_cls: type[Self],
        model: "AbstractModel",
        data: Any,
        **kwargs: Any,
    ) -> TargetT:
        """Abstract method to create a target from raw data for fitting."""

    def _equals(self, other: object) -> bool:
        return super()._equals(other) and eqx.tree_equal(self.target, other.target)

    def init_params(self, *args: Any, **kwargs: Any) -> Self:  # noqa
        """Initialize model parameters for fitting."""
        params = jax.tree.map(
            lambda p: self._initialize_param(self.model, p, self.target),
            self.model.parameters,
            is_leaf=lambda x: isinstance(x, AbstractParameter),
        )
        return self.replace(model=self.model.replace(parameters=params))

    @staticmethod
    @dispatch.abstract
    def _initialize_param(
        model: "AbstractModel",
        param: AbstractParameter,
        target: AbstractFitTarget,
    ) -> AbstractParameter:
        """Abstract method to initialize model parameters for fitting."""
