from abc import abstractmethod
from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any, Self, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp

from grgg._typing import Reals
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


ComputeKwargsT = tuple[RandomGenerator, dict[str, Any], dict[str, Any]]


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

    def moment(self, n: int, *args: Any, **kwargs: Any) -> Reals:
        """Compute the n-th moment of the statistic.

        Raises
        ------
        ValueError
            If the n-th moment is not supported.
        """
        if n not in self.supported_moments:
            raise self.unsupported_moment_exception(n)
        moment = self._moment(n, *args, **kwargs)
        return self.postprocess(moment)

    def postprocess(self, moment: Reals) -> Reals:
        """Post-process the computed moment, e.g., to ensure non-negativity.

        This is mostly useful for sampling-based estimates that may sometimes produce
        results that are out of theoretical bounds.
        """
        return moment

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

    def _equals(self, other: object) -> bool:
        return (
            super()._equals(other)
            and self.label == other.label
            and self.supported_moments == other.supported_moments
            and self.module.equals(other.module)
        )

    def prepare_compute_kwargs(self, **kwargs: Any) -> ComputeKwargsT:
        """Split kwargs into those for :class:`grgg.utils.compute.MapReduce`
        and those for other purposes and configure the random generator appropriately,
        so it may be distributed across many computation loops.

        Returns
        -------
        rng
            A random generator or `None`.
        mr_kwargs
            Keyword arguments for :class:`grgg.utils.compute.MapReduce`.
        loop_kwargs
            Keyword arguments for outer loops, usually :func:`jax.lax.map`.
        """
        mr_kwargs, loop_kwargs = split_kwargs_by_signature(MapReduce, **kwargs)
        rng = RandomGenerator.from_seed(mr_kwargs.pop("rng", None))
        if self.use_sampling:
            # Make sure each sampling call gets a different key
            # and the original rng is not used again, but mutated
            # a single time.
            rng = rng.child
        loop_kwargs["batch_size"] = self.model._get_batch_size(
            loop_kwargs.get("batch_size")
        )
        return rng, mr_kwargs, loop_kwargs

    def split_compute_kwargs(
        self, n: int = 2, *, same_seed: bool = False, **kwargs: Any
    ) -> tuple[dict[str, Any], ...]:
        """Split compute kwargs for using in multiple routines.

        Parameters
        ----------
        n
            Number of splits.
        same_seed
            Whether to keep the same PRNG seed for all splits.
        **kwargs
            Keyword arguments to split.
        """
        rng, kwargs, _ = self.prepare_compute_kwargs(**kwargs)
        if rng is None and same_seed:
            rng = RandomGenerator()
        if rng is not None:
            if same_seed:
                rngs = tuple(rng.copy() for _ in range(n))
            else:
                rngs = RandomGenerator(rng).split(n)
            return tuple({**kwargs, "rng": rng} for rng in rngs)
        return tuple(kwargs.copy() for _ in range(n))


class AbstractErgmStatistic[MT](AbstractStatistic[MT]):
    """Abstract base class for ERGM statistics."""

    n_samples: int = eqx.field(static=True, default=0, kw_only=True, converter=int)
    _importance_weights: eqx.AbstractVar[Reals]

    @property
    def use_sampling(self) -> bool:
        return self.model.is_heterogeneous and self.n_samples > 0

    @property
    def importance_weights(self) -> Reals:
        """Importance sampling weights."""
        return jax.lax.stop_gradient(self._importance_weights)

    def moment(
        self, n: int, *args: Any, n_samples: int = 0, repeat: int = 1, **kwargs: Any
    ) -> Reals:
        """Compute the n-th moment of the statistic for the nodes."""
        if n_samples > 0:
            return self.replace(n_samples=n_samples).moment(
                n, *args, repeat=repeat, **kwargs
            )
        if not self.use_sampling or repeat <= 1:
            return super().moment(n, *args, **kwargs)
        moments = jnp.stack(
            [self.moment(n, *args, repeat=1, **kwargs) for _ in range(repeat)]
        )
        return moments

    def prepare_compute_kwargs(self, **kwargs: Any) -> ComputeKwargsT:
        rng, mr_kwargs, loop_kwargs = super().prepare_compute_kwargs(**kwargs)
        mr_kwargs = {"n_samples": self.n_samples, **mr_kwargs}
        return rng, mr_kwargs, loop_kwargs


class AbstractErgmViewStatistic[QT](AbstractErgmStatistic[QT]):
    """Abstract base class for ERGM statistics on model views."""

    module: eqx.AbstractVar[QT]

    @property
    def view(self) -> QT:
        return self.module

    def moment(self, n: int, *args: Any, **kwargs: Any) -> Reals:
        """Compute the n-th moment of the statistic for the nodes."""
        m = super().moment(n, *args, **kwargs)
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

    @property
    def _importance_weights(self) -> Reals:
        return self.model.nodes.degree() if self.use_sampling else jnp.array(1.0)


class AbstractErgmNodeLocalStructureStatistic[VT](AbstractErgmNodeStatistic[VT]):
    """Abstract base class for node local structure statistics."""

    def postprocess(self, moment: Reals) -> Reals:
        return jnp.clip(moment, 0, jnp.inf)


class AbstractErgmNodePairStatistic[ET](AbstractErgmViewStatistic[ET]):
    """Abstract base class for node pair statistics."""

    module: eqx.AbstractVar["ET"]

    @property
    def pairs(self) -> "ET":
        return self.module

    @property
    def nodes(self) -> "VT":
        return self.model.nodes
