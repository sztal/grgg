from typing import Any, TypeVar

import equinox as eqx

from grgg.models.abc import AbstractModel, AbstractModelFunctions

from .sampling import ErgmSample
from .views import AbstractErgmNodePairView, AbstractErgmNodeView

__all__ = ("AbstractErgm",)


V = TypeVar("V", bound=AbstractErgmNodeView)
E = TypeVar("E", bound=AbstractErgmNodePairView)
F = TypeVar("F", bound=AbstractModelFunctions)


class AbstractErgm[V, E, F](AbstractModel[F]):
    """Abstract base class for ERGMs."""

    n_nodes: eqx.AbstractVar[int]

    is_directed: eqx.AbstractClassVar[bool]
    nodes_cls: eqx.AbstractClassVar[type[V]]
    pairs_cls: eqx.AbstractClassVar[type[E]]

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
    def nodes(self) -> V:
        """Nodes view."""
        return self.nodes_cls(self)

    @property
    def pairs(self) -> E:
        """Node pairs view."""
        return self.pairs_cls(self)

    def sample(self, *args: Any, **kwargs: Any) -> ErgmSample:
        """Sample from the model."""
        return self.nodes.sample(*args, **kwargs)
