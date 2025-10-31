from abc import abstractmethod
from typing import Any

import equinox as eqx
import jax.numpy as jnp

from grgg._typing import Reals
from grgg.models.abc import AbstractModel

from .sampling import ErgmSample
from .views import AbstractErgmNodePairView, AbstractErgmNodeView

__all__ = ("AbstractErgm",)


class AbstractErgm(AbstractModel):
    """Abstract base class for ERGMs."""

    n_nodes: eqx.AbstractVar[int]
    is_directed: eqx.AbstractClassVar[bool]
    nodes_cls: eqx.AbstractClassVar[type[AbstractErgmNodeView]]
    pairs_cls: eqx.AbstractClassVar[type[AbstractErgmNodePairView]]

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

    @abstractmethod
    def free_energy(self, *args: Any, **kwargs: Any) -> Reals:
        """Compute the free energy of the model."""

    def partition_function(self, *args: Any, **kwargs: Any) -> Reals:
        """Compute the partition function of the model."""
        return jnp.exp(-self.free_energy(*args, **kwargs))
