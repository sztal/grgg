from abc import abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, Self

import equinox as eqx
import jax.numpy as jnp

from grgg._options import options
from grgg.abc import AbstractModule
from grgg.utils import parse_switch_flag

from ._views import NodePairView, NodeView

__all__ = ("AbstractModelModule",)

ParamsT = Mapping[str, jnp.ndarray]


class AbstractModelModule(AbstractModule):
    """Abstract base class for model modules."""

    n_nodes: eqx.AbstractVar[int]

    @property
    @abstractmethod
    def n_units(self) -> int:
        """Number of units in the model."""

    @property
    @abstractmethod
    def parameters(self) -> ParamsT | Sequence[ParamsT]:
        """Model parameters."""

    @property
    @abstractmethod
    def is_heterogeneous(self) -> bool:
        """Whether the module has heterogeneous parameters."""

    @property
    def is_homogeneous(self) -> bool:
        """Whether the module has homogeneous parameters."""
        return not self.is_heterogeneous

    @property
    @abstractmethod
    def is_quantized(self) -> bool:
        """Whether the module has quantized parameters."""

    @property
    def nodes(self) -> NodeView:
        """Nodes view."""
        return NodeView(self)

    @property
    def pairs(self) -> NodePairView:
        """Node pairs view."""
        return NodePairView(self)

    @abstractmethod
    def set_parameters(self, parameters: ParamsT | Sequence[ParamsT]) -> Self:
        """Get shallow copy with update parameter values."""

    def _preprocess_inputs(self, *inputs: jnp.ndarray) -> jnp.ndarray:
        """Preprocess inputs to ensure it is all :mod:`jax` arrays."""
        return tuple(jnp.asarray(input) for input in inputs)

    def _get_batch_size(self, value: int | None = None) -> int:
        """Get batch size from value or options."""
        if value is None:
            value = int(options.batch.size)
        if value <= 0:
            value = self.n_nodes
        return int(value)

    def _get_progress(self, value: bool | None = None) -> tuple[bool, dict[str, Any]]:
        """Get progress value from value or options."""
        if value is None:
            value = self.n_nodes >= options.batch.auto_progress
        value, opts = parse_switch_flag(value)
        return value, opts


class AbstractModel(AbstractModelModule):
    """Abstract base class for models."""

    __grgg_model__ = True
