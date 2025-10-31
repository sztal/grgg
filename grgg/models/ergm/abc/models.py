from typing import Any

import equinox as eqx

from grgg.models.abc import AbstractModel

from .sampling import ErgmSample
from .views import AbstractErgmNodePairView, AbstractErgmNodeView

__all__ = ("AbstractErgm",)


class AbstractErgm(AbstractModel):
    """Abstract base class for ERGMs."""

    n_nodes: eqx.AbstractVar[int]
    nodes: eqx.AbstractVar[AbstractErgmNodeView]
    pairs: eqx.AbstractVar[AbstractErgmNodePairView]

    is_directed: eqx.AbstractClassVar[bool]
    nodes_cls: eqx.AbstractClassVar[type[AbstractErgmNodeView]]
    pairs_cls: eqx.AbstractClassVar[type[AbstractErgmNodePairView]]

    def __check_init__(self) -> None:
        if self.n_nodes <= 0:
            errmsg = f"'n_nodes' must be positive, got {self.n_nodes}."
            raise ValueError(errmsg)

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "nodes", self.nodes_cls(self))
        object.__setattr__(self, "pairs", self.pairs_cls(self))

    @property
    def n_units(self) -> int:
        """Number of units in the model."""
        return self.n_nodes

    @property
    def is_undirected(self) -> bool:
        """Whether the model is undirected."""
        return not self.is_directed

    def sample(self, *args: Any, **kwargs: Any) -> ErgmSample:
        """Sample from the model."""
        return self.nodes.sample(*args, **kwargs)
