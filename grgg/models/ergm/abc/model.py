from collections.abc import Callable
from functools import singledispatchmethod
from types import NoneType
from typing import Any, ClassVar, NamedTuple

import equinox as eqx

from grgg._typing import Real, Reals
from grgg.models.abc import AbstractModel
from grgg.statistics.abc import AbstractErgmNodePairStatistic, AbstractErgmNodeStatistic
from grgg.statistics.motifs import (
    AbstractErgmNodeMotifStatistic,
    AbstractErgmNodePairMotifStatistic,
)
from grgg.utils.misc import split_kwargs_by_signature

from .functions import AbstractErgmFunctions
from .optimize import ErgmOptimizer
from .sampling import ErgmSample
from .views import AbstractErgmNodePairView, AbstractErgmNodeView

__all__ = ("AbstractErgm",)

LagrangianT = Callable[["AbstractErgm", Any, ...], Real]


class AbstractErgm(AbstractModel):
    """Abstract base class for ERGMs."""

    n_nodes: eqx.AbstractVar[int]

    is_directed: eqx.AbstractClassVar[bool]
    functions: eqx.AbstractClassVar[type[AbstractErgmFunctions]]

    nodes_cls: eqx.AbstractClassVar[type[AbstractErgmNodeView]]
    pairs_cls: eqx.AbstractClassVar[type[AbstractErgmNodePairView]]
    optimizer_cls: ClassVar[type[ErgmOptimizer]] = ErgmOptimizer

    def __check_init__(self) -> None:
        if self.n_nodes <= 0:
            errmsg = f"'n_nodes' must be positive, got {self.n_nodes}."
            raise ValueError(errmsg)

    @property
    def n_units(self) -> int:
        """Number of units in the model."""
        return self.n_nodes

    @property
    def is_undirected(self) -> bool:
        """Whether the model is undirected."""
        return not self.is_directed

    @property
    def nodes(self) -> AbstractErgmNodeView:
        """Node view of the model."""
        return self.nodes_cls(self)

    @property
    def pairs(self) -> AbstractErgmNodePairView:
        """Node pair view of the model."""
        return self.pairs_cls(self)

    def sample(self, *args: Any, **kwargs: Any) -> ErgmSample:
        """Sample from the model."""
        return self.nodes.sample(*args, **kwargs)

    # Model functions ----------------------------------------------------------------

    def free_energy(self, *args: Any, **kwargs: Any) -> Reals:
        """Compute the free energy of the model."""
        return self.functions.free_energy(self, *args, **kwargs)

    def partition_function(self, *args: Any, **kwargs: Any) -> Reals:
        """Compute the partition function of the model."""
        return self.functions.partition_function(self, *args, **kwargs)

    def param_statistic(self, name: str, obj: Any, **kwargs: Any) -> Reals:
        """Compute the sufficient statistic for a given parameter based on object."""
        param = getattr(self.parameters, name)
        if issubclass(param.statistic, AbstractErgmNodeStatistic):
            statfun = getattr(self.nodes, param.statistic.label)
        elif issubclass(param.statistic, AbstractErgmNodePairStatistic):
            statfun = getattr(self.pairs, param.statistic.label)
        elif issubclass(param.statistic, AbstractErgmNodeMotifStatistic):
            statfun = getattr(self.nodes.motifs, param.statistic.label)
        elif issubclass(param.statistic, AbstractErgmNodePairMotifStatistic):
            statfun = getattr(self.pairs.motifs, param.statistic.label)
        else:
            statfun = getattr(self, param.statistic.label)
        kwargs, _ = split_kwargs_by_signature(statfun.observed, **kwargs)
        return statfun.observed(obj, **kwargs)

    def sufficient_statistics(
        self, obj: Any | None = None, **kwargs: Any
    ) -> NamedTuple:
        """Compute the sufficient statistics of the model."""
        return self._sufficient_statistics(obj, **kwargs)

    @singledispatchmethod
    def _sufficient_statistics(self, obj: Any, **kwargs: Any) -> NamedTuple:
        stats = {}
        for name in self.Parameters.names:
            stats[name] = self.param_statistic(name, obj, **kwargs)
        return self._sufficient_statistics(None, **stats)

    @_sufficient_statistics.register
    def _(self, _: NoneType, **stats: Any) -> NamedTuple:
        return self.Parameters(*(stats[name] for name in self.Parameters.names))

    def hamiltonian(self, obj: Any, **kwargs: Any) -> Real:
        """Compute the Hamiltonian of the model."""
        return self.functions.hamiltonian(self, obj, **kwargs)
