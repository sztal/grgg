from collections.abc import Callable
from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any, ClassVar, Self, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp

from grgg._options import options
from grgg._typing import Integers, Reals
from grgg.abc import AbstractModule
from grgg.utils.misc import split_kwargs
from grgg.utils.random import RandomGenerator

if TYPE_CHECKING:
    from grgg.models.abc import AbstractModel, AbstractModelModule
    from grgg.models.ergm.abc.model import AbstractErgm, E, P, Q, S, V

    T = TypeVar("T", bound=AbstractModel)
    M = TypeVar("M", bound=AbstractModelModule[T])
    TE = TypeVar("TE", bound=AbstractErgm[P, V, E, S])
    ME = TypeVar("ME", bound=AbstractModelModule[TE])

MT = TypeVar("MT", bound="ME")
QT = TypeVar("QT", bound="Q")
VT = TypeVar("VT", bound="V")
ET = TypeVar("ET", bound="E")

__all__ = (
    "AbstractStatistic",
    "AbstractErgmStatistic",
    "AbstractErgmViewStatistic",
    "AbstractErgmNodeStatistic",
    "AbstractErgmNodePairStatistic",
)


OptsT = dict[str, Any]
ComputeKwargsT = tuple[OptsT, ...]
MomentMethodT = Callable[..., Reals]


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

    def __init__(self, module: MT) -> None:
        self.module = module

    def __call__(self, **kwargs: Any) -> Self:
        """Compute first moment of the statistic."""
        return self.moment(1, **kwargs)

    @property
    def model(self) -> "T":
        return self.module.model

    def moment(self, n: int, **kwargs: Any) -> Reals:
        """Compute the n-th moment of the statistic.

        Parameters
        ----------
        n
            The order of the moment.
        **kwargs
            Additional keyword arguments.

        Raises
        ------
        NotImplementedError
            If the n-th moment is not supported.
        """
        if kwargs:
            self = self.replace(**kwargs)
        method = self._get_moment_method(n)
        return self.postprocess(method())

    def postprocess(self, moment: Reals) -> Reals:
        """Post-process the computed moments."""
        return moment

    def _get_moment_method(self, n: int) -> MomentMethodT:
        """Get the appropriate moment method."""
        method_type = "heterogeneous" if self.model.is_heterogeneous else "homogeneous"
        try:
            method = getattr(self, f"_get_{method_type}_method")(n)
        except AttributeError as exc:
            raise self.unsupported_moment_exception(n) from exc
        return eqx.filter_jit(method)

    def expectation(self, *args: Any, **kwargs: Any) -> Reals:
        """Compute the expectation of the statistic."""
        return self.moment(1, *args, **kwargs)

    def variance(self, *args: Any, **kwargs: Any) -> Reals:
        """Compute the variance of the statistic."""
        return self.moment(2, *args, **kwargs) - self.expectation(*args, **kwargs) ** 2

    def unsupported_moment_exception(self, n: int) -> NotImplementedError:
        cn = self.__class__.__name__
        method_type = "heterogeneous" if self.model.is_heterogeneous else "homogeneous"
        errmsg = f"'{cn}' does not support {method_type} moment of order {n}"
        return NotImplementedError(errmsg)

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

    def _get_homogeneous_method(self, n: int) -> MomentMethodT:
        return getattr(self, f"_homogeneous_m{n}")

    def _get_heterogeneous_method(self, n: int) -> MomentMethodT:
        return getattr(self, f"_heterogeneous_m{n}")


class AbstractErgmStatistic[MT](AbstractStatistic[MT]):
    batch_size: int | None = eqx.field(static=True)
    unroll: int = eqx.field(static=True)
    mc: int | bool = eqx.field(static=True)
    repeat: int = eqx.field(static=True)
    average: bool = eqx.field(static=True)
    same_seed: bool = eqx.field(static=True)
    key: Integers | None

    supports_monte_carlo: ClassVar[bool] = True

    def __init__(
        self,
        module: MT,
        *,
        key: Integers | None = None,
        rng: Integers | None = None,
        **kwargs: Any,
    ) -> None:
        """ERGM statistic.

        Parameters
        ----------
        batch_size
            The batch size for the main computation loop.
        unroll
            The unroll factor for the inner reduction loops.
            In general, should be rather small. The default value is typically best.
        mc
            Number of Monte Carlo samples to use for estimates.
            If `False`, exact computation is used where possible.
            If `True`, a default number of samples is chosen.
            If `None`, the automatic choice is made based on the model size.
        repeat
            Number of independent repetitions for Monte Carlo estimates.
        average
            Whether to average over repetitions for Monte Carlo estimates.
            If `False`, the estimates are stacked along the first axis.
        same_seed
            Whether to use the same random seed for coupled computations,
            e.g. when estimating triangle and t-wedge counts for computing
            the clustering coefficient.
        key, rng
            The random key (or generator) to use for sampling-based estimates.
            `rng` is an alias for `key`.
        """
        if not self.supports_monte_carlo:
            if (field := "mc") in kwargs:
                cn = self.__class__.__name__
                errmsg = (
                    f"'{cn}' does not support Monte Carlo sampling "
                    f"(got '{field}={kwargs[field]}')"
                )
                raise NotImplementedError(errmsg)
            kwargs[field] = False
        super().__init__(module)
        loop_kwargs, mc_kwargs = split_kwargs(options.loop.__annotations__, **kwargs)
        loop_opts = options.loop.replace(**loop_kwargs)
        self.batch_size = loop_opts.batch_size
        self.unroll = loop_opts.unroll
        mc_opts = options.monte_carlo.from_size(self.model.n_units, **mc_kwargs)
        self.mc = mc_opts.mc
        self.repeat = mc_opts.repeat
        self.average = mc_opts.average
        self.same_seed = mc_opts.same_seed
        if key is None and rng is not None:
            key = rng
        if isinstance(key, RandomGenerator):
            key = key.child
        self.key = RandomGenerator.make_key(key) if self.use_mc else None

    @property
    def use_mc(self) -> bool:
        """Whether to use Monte Carlo sampling for estimates."""
        return bool(self.mc)

    def moment(self, n: int, **kwargs: Any) -> Reals:
        if kwargs:
            self = self.replace(**kwargs)
        if not self.use_mc:
            return super().moment(n)
        if self.average:
            if self.repeat == 1:
                return super().moment(n)
            moment = 0.0
            for r in range(self.repeat):
                key = jax.random.fold_in(RandomGenerator.make_key(self.key), r)
                moment += super().moment(n, key=key) / self.repeat
            return moment
        moments = []
        for r in range(self.repeat):
            key = jax.random.fold_in(RandomGenerator.make_key(self.key), r)
            moment = super().moment(n, key=key)
            moments.append(moment)
        return jnp.stack(moments)

    def split_options(self, n: int = 1, **kwargs: Any) -> tuple[OptsT, ...]:
        if not self.use_mc:
            return tuple({"batch_size": self.batch_size} for _ in range(n))
        opts = {
            "batch_size": self.batch_size,
            "unroll": self.unroll,
            "mc": self.mc,
            "repeat": self.repeat,
            "average": self.average,
            "same_seed": self.same_seed,
            "key": self.key,
            **kwargs,
        }
        if self.same_seed:
            return tuple(opts.copy() for _ in range(n))
        return tuple({**opts, "key": jax.random.fold_in(self.key, i)} for i in range(n))

    def unsupported_moment_exception(self, n: int) -> NotImplementedError:
        cn = self.__class__.__name__
        method_type = "heterogeneous" if self.model.is_heterogeneous else "homogeneous"
        estimator = "Monte Carlo" if self.use_mc else "exact"
        errmsg = (
            f"'{cn}' does not support {method_type} {estimator} moment of order {n}"
        )
        return NotImplementedError(errmsg)

    def _get_heterogeneous_method(self, n: int) -> MomentMethodT:
        if self.use_mc:
            return getattr(self, f"_heterogeneous_m{n}_monte_carlo")
        return getattr(self, f"_heterogeneous_m{n}_exact")

    def observed(self, obj: Any, *args: Any, **kwargs: Any) -> Reals:
        """Compute the observed value of the statistic for a given object."""
        raise NotImplementedError

    @singledispatchmethod
    def check_observed(self, obj: Any) -> None:
        """Check if the given object is compatible with the statistic."""
        n = self.model.n_units
        try:
            if obj.shape != (n, n):
                errmsg = f"expected object of shape ({n}, {n}), got {obj.shape}"
                raise ValueError(errmsg)
        except AttributeError as exc:
            errmsg = f"object of type '{type(obj)}' is not supported"
            raise TypeError(errmsg) from exc

    try:
        import igraph as ig

        @check_observed.register
        def _(self, obj: ig.Graph) -> None:  # noqa
            n = self.model.n_units
            if obj.vcount() != n:
                errmsg = f"expected igraph Graph with {n} vertices, got {obj.vcount()}"
                raise ValueError(errmsg)

    except ImportError:
        pass

    try:
        import networkx as nx

        @check_observed.register
        def _(self, obj: nx.Graph) -> None:  # noqa
            n = self.model.n_units
            if obj.number_of_nodes() != n:
                errmsg = (
                    f"expected networkx Graph with {n} nodes, "
                    f"got {obj.number_of_nodes()}"
                )
                raise ValueError(errmsg)

    except ImportError:
        pass


class AbstractErgmViewStatistic[QT](AbstractErgmStatistic[QT]):
    module: eqx.AbstractVar[QT]

    @property
    def view(self) -> QT:
        return self.module

    def postprocess(self, moment: Reals) -> Reals:
        """Post-process the computed moment of the statistic for the nodes."""
        m = super().postprocess(moment)
        if self.model.is_homogeneous and self.view.is_active:
            return jnp.full(self.view.shape, m)
        return m


class AbstractErgmNodeStatistic[VT](AbstractErgmViewStatistic[VT]):
    module: eqx.AbstractVar["VT"]

    @property
    def nodes(self) -> "VT":
        return self.module

    @property
    def pairs(self) -> "ET":
        return self.model.pairs


class AbstractErgmNodePairStatistic[ET](AbstractErgmViewStatistic[ET]):
    module: eqx.AbstractVar["ET"]

    @property
    def pairs(self) -> "ET":
        return self.module

    @property
    def nodes(self) -> "VT":
        return self.model.nodes
