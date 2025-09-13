from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from functools import wraps
from typing import Any

import jax.numpy as np
from flax import nnx

from grgg._options import options
from grgg._typing import Floats
from grgg.abc import AbstractModule
from grgg.utils import parse_switch_flag

from ._views import NodePairView, NodeView
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
    def nodes(self) -> NodeView:
        """Nodes view."""
        return NodeView(self)

    @property
    def pairs(self) -> NodePairView:
        """Node pairs view."""
        return NodePairView(self)

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

    def get_batch_size(self, batch_size: int | None = None) -> int:
        """Get batch size from value or options."""
        if batch_size is None:
            batch_size = int(options.batch.size)
        if batch_size <= 0:
            batch_size = self.n_nodes
        return int(batch_size)

    def get_progress(self, progress: bool | None = None) -> tuple[bool, dict[str, Any]]:
        """Get progress value from value or options."""
        if progress is None:
            progress = self.n_nodes >= options.batch.auto_progress
        progress, opts = parse_switch_flag(progress)
        return progress, opts


class AbstractModel(AbstractModelModule):
    """Abstract base class for models."""

    __grgg_model__ = True
