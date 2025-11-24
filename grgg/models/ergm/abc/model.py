from collections.abc import Callable
from inspect import getmembers, isabstract
from typing import Any, ClassVar

import equinox as eqx

from grgg._typing import Real, Reals
from grgg.models.abc import AbstractModel
from grgg.statistics.abc import AbstractErgmStatistic

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

    fit_default_method: ClassVar[str] = "lagrangian"

    supported_statistics: eqx.AbstractClassVar[dict[str, str]]

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        stats = {}
        if not isabstract(cls):
            for typ in (
                cls,
                cls.nodes_cls,
                cls.nodes_cls.motifs_cls,
                cls.pairs_cls,
                cls.pairs_cls.motifs_cls,
            ):
                for _, member in getmembers(
                    typ,
                    lambda x: (
                        isinstance(x, type) and issubclass(x, AbstractErgmStatistic)
                    ),
                ):
                    if member.label in stats:
                        errmsg = (
                            f"Duplicate statistic label '{member.label}' found "
                            f"in '{stats[member.label].__qualname__}' and "
                            f"'{member.__qualname__}'. Statistic labels must be unique."
                        )
                        raise ValueError(errmsg)
                    stats[member.label] = member.namespace
            cls.supported_statistics = stats

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

    def get_statistic(self, label: str) -> AbstractErgmStatistic:
        """Get a statistic by its label."""
        try:
            namespace = self.supported_statistics.get(label)
        except KeyError as exc:
            cn = self.__class__.__name__
            errmsg = f"'{label}' statistic is not supported by '{cn}'"
            raise ValueError(errmsg) from exc
        obj = self
        for part in (*namespace.split("."), label):
            obj = getattr(obj, part)
        return obj

    # Sampling -----------------------------------------------------------------------

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

    def hamiltonian(self, obj: Any, **kwargs: Any) -> Real:
        """Compute the Hamiltonian of the model."""
        return self.functions.hamiltonian(self, obj, **kwargs)

    def lagrangian(self, obj: Any, **kwargs: Any) -> Real:
        """Compute the Lagrangian of the model."""
        return self.functions.lagrangian(self, obj, **kwargs)
