from abc import abstractmethod
from collections.abc import Mapping
from typing import Any, Self

import equinox as eqx
import jax.numpy as jnp

from grgg._options import options
from grgg.utils.misc import parse_switch_flag

from .modules import AbstractModelModule
from .parameters import AbstractParameter

__all__ = ("AbstractModel",)


class AbstractModel(AbstractModelModule[Self]):
    """Abstract base class for models."""

    n_units: eqx.AbstractVar[int]

    def __check_init__(self) -> None:
        if self.n_units <= 0:
            errmsg = f"'n_units' must be positive, got {self.n_units}."
            raise ValueError(errmsg)
        for name, parameter in self.parameters.items():
            if not parameter.is_scalar and len(parameter) != self.n_units:
                errmsg = (
                    f"all non-scalar parameters must have leading axis size equal to "
                    f"'n_units' ({self.n_units}), but parameter '{name}' has "
                    f"{parameter.size} instead."
                )
                raise ValueError(errmsg)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._repr_inner()})"

    @property
    def model(self) -> Self:
        """Self as model."""
        return self

    @property
    def is_heterogeneous(self) -> bool:
        """Whether the model has heterogeneous parameters."""
        return all(param.is_heterogeneous for param in self.parameters.values())

    @property
    def is_homogeneous(self) -> bool:
        """Whether the model has homogeneous parameters."""
        return not self.is_heterogeneous

    @property
    def is_quantized(self) -> bool:
        """Whether the model is quantized."""
        return False

    @property
    @abstractmethod
    def parameters(self) -> Mapping[str, AbstractParameter]:
        """Parameters of the model."""

    @abstractmethod
    def _repr_inner(self) -> str:
        """Inner part of the string representation."""
        return f"{self.n_units}"

    def _preprocess_inputs(self, *inputs: jnp.ndarray) -> jnp.ndarray:
        return tuple(jnp.asarray(input) for input in inputs)

    def _get_progress(self, value: bool | None = None) -> tuple[bool, dict[str, Any]]:
        """Get progress value from value or options."""
        if value is None:
            value = self.n_nodes >= options.auto.progress
        value, opts = parse_switch_flag(value)
        return value, opts

    @abstractmethod
    def sample(self, *args: Any, **kwargs: Any) -> Any:
        """Sample from the model."""
