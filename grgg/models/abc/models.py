from abc import abstractmethod
from typing import Any, TypeVar

import equinox as eqx
import jax.numpy as jnp

from grgg._options import options
from grgg.utils.misc import parse_switch_flag

from .modules import AbstractModelModule
from .parameters import AbstractParameters
from .sampling import AbstractModelSampler

__all__ = ("AbstractModel",)


T = TypeVar("T", bound="AbstractModel")
P = TypeVar("M", bound="AbstractParameters")
S = TypeVar("S", bound=AbstractModelSampler)


class AbstractModel[T, P, S](AbstractModelModule[T]):
    """Abstract base class for models."""

    parameters: eqx.AbstractVar[P]
    n_units: eqx.AbstractVar[int]

    parameters_cls: eqx.AbstractClassVar[type[P]]
    sampler_cls: eqx.AbstractClassVar[type[S]]

    def __check_init__(self) -> None:
        if self.n_units <= 0:
            errmsg = f"'n_units' must be positive, got {self.n_units}."
            raise ValueError(errmsg)
        if not isinstance(self.parameters, self.parameters_cls):
            errmsg = (
                f"'parameters' must be an instance of '{self.parameters_cls.__name__}',"
                f" got '{type(self.parameters).__name__}' instead."
            )
            raise TypeError(errmsg)
        for parameter in self.parameters:
            if not parameter.is_scalar and len(parameter) != self.n_units:
                name = parameter.name
                errmsg = (
                    f"all non-scalar parameters must have leading axis size equal to "
                    f"'n_units' ({self.n_units}), but parameter '{name}' has "
                    f"{parameter.size} instead."
                )
                raise ValueError(errmsg)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._repr_inner()})"

    @property
    def model(self) -> T:
        """Self as model."""
        return self

    @property
    @abstractmethod
    def is_heterogeneous(self) -> bool:
        """Whether the model has heterogeneous parameters."""

    @property
    def is_homogeneous(self) -> bool:
        """Whether the model has homogeneous parameters."""
        return not self.is_heterogeneous

    @property
    def is_quantized(self) -> bool:
        """Whether the model is quantized."""
        return False

    @abstractmethod
    def _repr_inner(self) -> str:
        """Inner part of the string representation."""
        return f"{self.n_units}"

    def _preprocess_inputs(self, *inputs: jnp.ndarray) -> jnp.ndarray:
        return tuple(jnp.asarray(input) for input in inputs)

    def _get_batch_size(self, value: int | None = None) -> int:
        """Get batch size from value or options."""
        if value is None:
            value = int(options.batch.size)
        if value <= 0:
            value = self.n_units
        return int(value)

    def _get_progress(self, value: bool | None = None) -> tuple[bool, dict[str, Any]]:
        """Get progress value from value or options."""
        if value is None:
            value = self.n_nodes >= options.batch.auto_progress
        value, opts = parse_switch_flag(value)
        return value, opts

    @property
    @abstractmethod
    def sampler(self) -> S:
        """Sampler for the model."""

    def sample(self, *args: Any, **kwargs: Any) -> Any:
        """Sample from the model."""
        return self.sampler.sample(*args, **kwargs)
