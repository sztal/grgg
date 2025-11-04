from collections.abc import Mapping
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import DTypeLike

from grgg._typing import Vector
from grgg.abc import AbstractModule
from grgg.utils.indexing import Shaped
from grgg.utils.misc import format_array

from .constraints import Constraint

__all__ = ("ParameterInfo", "Parameter", "ModelParameters")


class ParameterInfo(Shaped):
    """Information about a model parameter.

    Attributes
    ----------
    name
        Name of the parameter.
    shape
        Shape of the parameter.
    dtype
        Data type of the parameter.
    loc
        Location (start and end indices) of the parameter in a flattened array
        of model parameters.
    """

    loc: tuple[int, int] = eqx.field(static=True)
    shape: tuple[int, ...] = eqx.field(static=True)
    dtype: DTypeLike = eqx.field(
        static=True, default=eqx.jax.dtypes.canonicalize_dtype(float).type
    )
    name: str | None = eqx.field(static=True, default=None)
    constraints: tuple[Constraint, ...] = eqx.field(static=True, default=())

    def __check_init__(self) -> None:
        lsize = self.loc[1] - self.loc[0]
        if lsize < 1:
            errmsg = f"invalid location {self.loc}"
            raise ValueError(errmsg)
        if self.size != lsize:
            errmsg = (
                f"location {self.loc} inconsistent with shape {self.shape} "
                f"(size {self.size})"
            )
            raise ValueError(errmsg)

    def validate(self, data: jnp.ndarray) -> jnp.ndarray:
        """Validate parameter against specification."""
        if data.shape != self.shape:
            data = data.reshape(self.shape)
        if data.dtype != self.dtype:
            data = data.astype(self.dtype)
        for constraint in self.constraints:
            constraint.check(data, name=self.name)
        return data

    def make_param(self, data: jnp.ndarray) -> "Parameter":
        """Create a Parameter instance from data."""
        data = data[slice(*self.loc)]
        return Parameter(self.validate(data), self.name)

    def _equals(self, other: object) -> bool:
        return (
            super()._equals(other)
            and self.loc == other.loc
            and self.shape == other.shape
            and self.dtype == other.dtype
            and self.name == other.name
            and self.constraints == other.constraints
        )


class Parameter(Shaped):
    """Model parameter.

    Attributes
    ----------
    data
        Parameter value(s).
    name
        Name of the parameter.
    """

    def _equals(self, other: object) -> bool:
        return (
            super()._equals(other)
            and self.name == other.name
            and jnp.array_equal(self.data, other.data)
        )

    def __repr__(self) -> str:
        if self.is_scalar:
            return f"{self.__class__.__name__}({self.data.item()})"
        return f"{self.__class__.__name__}({format_array(self.data)})"

    def __getitem__(self, args: Any) -> jnp.ndarray:
        return self.replace(data=self.data[args])

    def __len__(self) -> int:
        return self.shape[0]

    @property
    def dtype(self) -> DTypeLike:
        """Data type of the parameter data."""
        return self.data.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the parameter data."""
        return self.data.shape

    @property
    def is_scalar(self) -> bool:
        """Whether the parameter is a scalar."""
        return hasattr(self.data, "ndim") and self.data.ndim == 0

    @property
    def is_homogeneous(self) -> bool:
        """Whether the parameter is homogeneous (all values identical)."""
        return self.is_scalar

    @property
    def is_heterogeneous(self) -> bool:
        """Whether the parameter is heterogeneous (not all values identical)."""
        return not self.is_homogeneous


class ModelParameters(AbstractModule, Mapping[str, Parameter]):
    """Model parameters.

    Attributes
    ----------
    data
        Parameter data as 1D array.
    info
        Information about the model parameters.
    """

    data: Vector
    _info: tuple[ParameterInfo, ...] = eqx.field(static=True)

    def __init__(
        self,
        data: Vector,
        *,
        _info: tuple[ParameterInfo, ...] = (),
        **param_infos: ParameterInfo,
    ) -> None:
        self.data = data
        self._info = _info + tuple(v.replace(name=k) for k, v in param_infos.items())

    def __check_init__(self) -> None:
        if min(pinfo.loc[0] for pinfo in self._info) != 0:
            errmsg = "parameter locations must start at 0"
            raise ValueError(errmsg)
        if max(pinfo.loc[1] for pinfo in self._info) != self.data.size:
            errmsg = f"parameter locations must end at data size {self.data.size}"
            raise ValueError(errmsg)
        if any(pinfo.name is None for pinfo in self._info):
            errmsg = "all parameters must be named"
            raise ValueError(errmsg)
        if self.data.size != sum(pinfo.size for pinfo in self._info):
            errmsg = (
                f"data size {self.data.size} inconsistent with parameter info "
                f"(expected size {sum(pinfo.size for pinfo in self._info)})"
            )
            raise ValueError(errmsg)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.param_reprs()})"

    def __getitem__(self, name: str) -> Parameter:
        return self.info[name].make_param(self.data)

    def __iter__(self):
        return (pinfo.name for pinfo in self._info)

    def __len__(self) -> int:
        return len(self._info)

    @property
    def info(self) -> dict[str, ParameterInfo]:
        """Information about the model parameters."""
        return {pinfo.name: pinfo for pinfo in self._info}

    def param_reprs(self) -> str:
        return ", ".join(repr(p) for p in self.values())
