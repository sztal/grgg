from typing import Any, TypeVar

import equinox as eqx

from grgg.models.abc import AbstractModel

from .sampling import AbstractErgmSampler, ErgmSample
from .views import AbstractErgmNodePairView, AbstractErgmNodeView, AbstractErgmView

__all__ = ("AbstractErgm",)


T = TypeVar("T", bound="AbstractErgm")
Q = TypeVar("Q", bound=AbstractErgmView)
V = TypeVar("V", bound=AbstractErgmNodeView)
E = TypeVar("E", bound=AbstractErgmNodePairView)
S = TypeVar("S", bound=AbstractErgmSampler)
X = TypeVar("X", bound=ErgmSample)


class AbstractErgm[T, V, E, S](AbstractModel[T, S]):
    """Abstract base class for ERGMs."""

    n_nodes: eqx.AbstractVar[int]

    is_directed: eqx.AbstractClassVar[bool]
    node_view_cls: eqx.AbstractClassVar[type[V]]
    pair_view_cls: eqx.AbstractClassVar[type[E]]

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
        return self.node_view_cls(self)

    @property
    def pairs(self) -> E:
        """Node pairs view."""
        return self.pair_view_cls(self)

    @property
    def sampler(self) -> S:
        return self.nodes.sampler

    def sample(self, *args: Any, **kwargs: Any) -> X:
        """Sample from the model."""
        return self.nodes.sample(*args, **kwargs)
