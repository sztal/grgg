from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from functools import wraps

import jax.numpy as np
from flax import nnx

from grgg._typing import Floats
from grgg.abc import AbstractModule

from ._indexing import NodeIndexer, NodePairIndexer
from .parameters import AbstractModelParameter

__all__ = ("AbstractModelModule",)

ParamsT = Mapping[str, AbstractModelParameter]


class AbstractModelModule(AbstractModule):
    """Abstract base class for model modules."""

    @property
    @abstractmethod
    def n_nodes(self) -> int:
        """Number of nodes in the model."""

    @property
    @abstractmethod
    def parameters(self) -> ParamsT | Sequence[ParamsT]:
        """Model parameters."""

    @property
    def nodes(self) -> NodeIndexer:
        """Node indexer."""
        return NodeIndexer(self)

    @property
    def pairs(self) -> NodePairIndexer:
        """Node pairs indexer."""
        return NodePairIndexer(self)

    @abstractmethod
    def define_function(self) -> Callable[[Floats, Floats, Floats], Floats]:
        """Define the module function."""

    def _define_function(self) -> Callable[[Floats, Floats, Floats], Floats]:
        function = self.define_function()

        @wraps(function)
        def wrapper(*args: Floats) -> Floats:
            args = tuple(np.asarray(a) for a in args)
            return function(*args)

        return nnx.jit(wrapper)
