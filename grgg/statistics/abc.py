from abc import abstractmethod
from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any, Self, TypeVar

import equinox as eqx
import jax.numpy as jnp

from grgg._typing import Integers, Reals
from grgg.abc import AbstractModule
from grgg.utils.compute import MapReduce
from grgg.utils.misc import split_kwargs_by_signature
from grgg.utils.random import RandomGenerator

if TYPE_CHECKING:
    from grgg.models.abc import AbstractModel, AbstractModelModule
    from grgg.models.ergm.abc.models import AbstractErgm, E, P, Q, S, V

    T = TypeVar("T", bound=AbstractModel)
    M = TypeVar("M", bound=AbstractModelModule[T])
    TE = TypeVar("TE", bound=AbstractErgm[P, V, E, S])
    ME = TypeVar("ME", bound=AbstractModelModule[TE])

MT = TypeVar("MT", bound="ME")
QT = TypeVar("QT", bound="Q")
VT = TypeVar("VT", bound="V")
ET = TypeVar("ET", bound="E")

__all__ = ("AbstractStatistic",)


ComputeKwargsT = tuple[bool, Integers, dict[str, Any], dict[str, Any]]


class AbstractStatistic[MT](AbstractModule):
    """Abstract base class for statistics.

    Attributes
    ----------
    model
        The model statistics is computed for.
    label
        The label of the statistic.
    """

    module: eqx.AbstractVar[MT]

    label: eqx.AbstractClassVar[str]
    supported_moments: eqx.AbstractClassVar[tuple[int, ...]]

    def __init_subclass__(cls) -> None:
        for moment in getattr(cls, "supported_moments", ()):
            if not isinstance(moment, int) or moment < 1:
                errmsg = "supported_moments must be a tuple of positive integers"
                raise ValueError(errmsg)
            for variant in ("homogeneous", "heterogeneous"):
                method_name = f"_{variant}_m{moment}"
                if not hasattr(cls, method_name):

                    def _moment_method(self, *args: Any, **kwargs: Any) -> Reals:  # noqa
                        pass

                    _moment_method.__name__ = method_name
                    setattr(cls, method_name, abstractmethod(_moment_method))

    def __call__(self, *args: Any, **kwargs: Any) -> Self:
        """Compute first moment of the statistic."""
        return self.moment(1, *args, **kwargs)

    @property
    def model(self) -> "T":
        return self.module.model

    @singledispatchmethod
    @classmethod
    @abstractmethod
    def from_module(cls, module: "T", *args: Any, **kwargs: Any) -> Self:
        """Create a statistic from a model."""
        raise cls.unsupported_module_exception(module)

    @abstractmethod
    def moment(self, n: int, *args: Any, **kwargs: Any) -> Reals:
        """Compute the n-th moment of the statistic.

        Raises
        ------
        ValueError
            If the n-th moment is not supported.
        """

    def _moment(self, n: int, *args: Any, **kwargs: Any) -> Reals:
        """Compute the n-th moment of the statistic for the model."""
        method = (
            getattr(self, f"_homogeneous_m{n}")
            if self.model.is_homogeneous
            else getattr(self, f"_heterogeneous_m{n}")
        )
        return method(*args, **kwargs)

    def expectation(self, *args: Any, **kwargs: Any) -> Reals:
        """Compute the expectation of the statistic."""
        return self.moment(1, *args, **kwargs)

    def variance(self, *args: Any, **kwargs: Any) -> Reals:
        """Compute the variance of the statistic."""
        return self.moment(2, *args, **kwargs) - self.expectation(*args, **kwargs) ** 2

    def unsupported_moment_exception(self, n: int) -> ValueError:
        errmsg = (
            f"{n}-th moment is not supported. "
            f"Supported moments: {self.supported_moments}"
        )
        return ValueError(errmsg)

    @classmethod
    def unsupported_module_exception(cls, module: object) -> TypeError:
        cn = cls.__name__
        errmsg = f"'{cn}' does not support module of type '{type(module)}'"
        return TypeError(errmsg)

    def equals(self, other: object) -> bool:
        return (
            super().equals(other)
            and self.label == other.label
            and self.supported_moments == other.supported_moments
            and self.module.equals(other.module)
        )

    def prepare_compute_kwargs(self, **kwargs: Any) -> ComputeKwargsT:
        """Split kwargs into those for :class:`grgg.utils.compute.MapReduce`
        and those for other purposes.
        """
        key = RandomGenerator.make_key(kwargs.pop("rng", None))
        mr_kwargs, other_kwargs = split_kwargs_by_signature(MapReduce, **kwargs)
        use_sampling = False
        return use_sampling, key, mr_kwargs, other_kwargs

    def split_compute_kwargs(
        self, n: int = 2, **kwargs: Any
    ) -> tuple[dict[str, Any], ...]:
        """Split kwargs into mutliple copies with different PRNG keys."""
        rng = kwargs.pop("rng", None)
        if rng is not None:
            rngs = RandomGenerator(rng).split(n)
            return tuple({**kwargs, "rng": rng} for rng in rngs)
        return tuple(kwargs.copy() for _ in range(n))


class AbstractErgmStatistic[MT](AbstractStatistic[MT]):
    """Abstract base class for ERGM statistics."""

    def prepare_compute_kwargs(self, **kwargs: Any) -> ComputeKwargsT:
        """Split kwargs into those for :class:`grgg.utils.compute.MapReduce`
        and those for :class:`grgg.utils.compute.Sampled`.

        The appropriate degree-based importance weights are added to the
        sampling kwargs if sampling is used and no weights are provided.
        """
        _, key, mr_kwargs, other_kwargs = super().prepare_compute_kwargs(**kwargs)
        # Set up weights for importance sampling if needed
        use_sampling = mr_kwargs.get("n_samples", -1) >= 0
        if use_sampling and "p" not in mr_kwargs:
            degree = self.model.nodes.degree()
            mr_kwargs["p"] = degree / degree.sum()
        return use_sampling, key, mr_kwargs, other_kwargs


class AbstractErgmViewStatistic[QT](AbstractErgmStatistic[QT]):
    """Abstract base class for ERGM statistics on model views."""

    module: eqx.AbstractVar[QT]

    @property
    def view(self) -> QT:
        return self.module

    def moment(self, n: int, *args: Any, **kwargs: Any) -> Reals:
        """Compute the n-th moment of the statistic for the nodes."""
        if n not in self.supported_moments:
            raise self.unsupported_moment_exception(n)
        m = self._moment(n, *args, **kwargs)
        if self.model.is_homogeneous and self.view.is_active:
            return jnp.full(self.view.shape, m)
        return m


class AbstractErgmNodeStatistic[VT](AbstractErgmViewStatistic[VT]):
    """Abstract base class for node statistics."""

    module: eqx.AbstractVar["VT"]

    @property
    def nodes(self) -> "VT":
        return self.module

    @property
    def pairs(self) -> "ET":
        return self.model.pairs


class AbstractErgmNodePairStatistic[ET](AbstractErgmViewStatistic[ET]):
    """Abstract base class for node pair statistics."""

    module: eqx.AbstractVar["ET"]

    @property
    def pairs(self) -> "ET":
        return self.module

    @property
    def nodes(self) -> "VT":
        return self.model.nodes
