from abc import abstractmethod
from typing import ClassVar, TypeVar

import equinox as eqx

from grgg._typing import Real, RealVector
from grgg.models.ergm.random_graphs.abc import AbstractRandomGraph, Mu

from .functions import UndirectedRandomGraphCoupling
from .sampling import AbstractUndirectedRandomGraphSampler
from .views import AbstractRandomGraphNodePairView, AbstractRandomGraphNodeView

__all__ = ("AbstractUndirectedRandomGraph",)


T = TypeVar("T", bound="AbstractUndirectedRandomGraph")
V = TypeVar("V", bound=AbstractRandomGraphNodeView)
E = TypeVar("E", bound=AbstractRandomGraphNodePairView)
S = TypeVar("S", bound=AbstractUndirectedRandomGraphSampler)


class AbstractUndirectedRandomGraph[T, V, E, S](AbstractRandomGraph[T, V, E, S]):
    """Abstract base class for undirected random graph models."""

    mu: eqx.AbstractVar[Mu]
    coupling: eqx.AbstractVar[UndirectedRandomGraphCoupling]

    is_directed: ClassVar[bool] = False

    @abstractmethod
    def __init__(
        self,
        n_nodes: int,
        mu: Real | RealVector | Mu | None = None,
    ) -> None:
        self.n_nodes = n_nodes
        self.mu = mu if isinstance(mu, Mu) else Mu(mu)
        self.coupling = self._init_coupling()

    def _repr_inner(self) -> str:
        return f"{self.n_nodes}, {self.mu}"

    @property
    def parameters(self) -> dict[str, Mu]:
        return {"mu": self.mu}

    def _equals(self, other: object) -> bool:
        return (
            super()._equals(other)
            and self.n_nodes == other.n_nodes
            and self.mu.equals(other.mu)
            and self.coupling.equals(other.coupling)
        )
