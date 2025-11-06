from typing import ClassVar, NamedTuple

import equinox as eqx

from grgg._typing import Real, RealVector
from grgg.models.abc import ParametersMeta
from grgg.models.ergm.random_graphs.abc import AbstractRandomGraph, Mu

from .functions import RandomGraphFunctions
from .views import RandomGraphNodePairView, RandomGraphNodeView

__all__ = ("RandomGraph",)


class RandomGraph(AbstractRandomGraph):
    """Undirected random graph model.

    It is equivalent to the `(n, p)`-Erdős–Rényi model when `mu` is homogeneous,
    or to the soft configuration model when `mu` is heterogeneous.

    Attributes
    ----------
    n_nodes
        Number of nodes.
    mu
        Parameter controlling the expected degree of nodes.
    """

    class Parameters(NamedTuple, metaclass=ParametersMeta):
        mu: RealVector

    n_nodes: int = eqx.field(static=True)
    mu: Mu

    is_directed: ClassVar[bool] = False
    functions: ClassVar[type[RandomGraphFunctions]] = RandomGraphFunctions

    nodes_cls: ClassVar[type[RandomGraphNodeView]] = RandomGraphNodeView
    pairs_cls: ClassVar[type[RandomGraphNodePairView]] = RandomGraphNodePairView

    def __init__(
        self,
        n_nodes: int,
        mu: Real | RealVector | Mu | None = None,
    ) -> None:
        self.n_nodes = n_nodes
        self.mu = mu if isinstance(mu, Mu) else Mu(mu)

    @property
    def parameters(self) -> "RandomGraph.Parameters":
        """Model parameters."""
        return self.Parameters(mu=self.mu)

    def _equals(self, other: object) -> bool:
        return super()._equals(other)
