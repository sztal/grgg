import weakref
from collections.abc import Callable
from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any, Self

import numpy as np

from . import options

if TYPE_CHECKING:
    from . import GRGG
    from .layers import AbstractGRGGLayer


IndexT = int | slice | np.ndarray


__all__ = ("Beta", "Mu")


class Outer:
    """Indexable outer operation.

    Attributes
    ----------
    v1
        First vector.
    v2
        Second vector.
    op
        Vectorized operation to apply.
    """

    def __init__(
        self,
        v1: np.ndarray,
        v2: np.ndarray | None = None,
        *,
        op: Callable[[np.ndarray, np.ndarray], np.ndarray] = np.multiply,
    ) -> None:
        self.v1 = np.asarray(v1)
        self.v2 = np.asarray(v2) if v2 is not None else self.v1
        self.op = op

    @singledispatchmethod
    def __getitem__(self, i: IndexT) -> np.ndarray:
        if isinstance(i, list):
            i = np.asarray(i)
            return self[i]
        out = self[i, i]
        if not np.isscalar(out):
            out = out[np.tril_indices_from(out, k=-1)]  # type: ignore
        return out

    @__getitem__.register
    def _(self, i: np.ndarray) -> np.ndarray:
        if i.ndim == 1:
            return self[i, i]
        if i.ndim == 2:
            vi = self.v1[i[:, 0]]
            vj = self.v2[i[:, 1]]
            return self.op(vi, vj)
        errmsg = "index array must be 1D or 2D"
        raise IndexError(errmsg)

    @__getitem__.register
    def _(self, i: tuple) -> np.ndarray | float:
        if not i:
            i = (slice(None), slice(None))
        if len(i) == 1:
            return self[i[0]]
        try:
            i, j = i
        except ValueError:
            errmsg = "wrong number indices"
            raise IndexError(errmsg) from None
        vi = self.v1[i]
        vj = self.v2[j]
        if not np.isscalar(vi):
            vi = vi[:, None]
        out = self.op(vi, vj)
        if out.size == 1:
            out = float(out.item())
        return out


class CouplingParameter:
    """Base class for coupling parameters.

    Attributes
    ----------
    value
        Parameter value.
    heterogeneous
        Whether the parameter is heterogeneous (varies across nodes).
    """

    def __init__(
        self,
        value: float | np.ndarray,
        *,
        heterogeneous: bool | None = None,
    ) -> None:
        self._value = None
        self._fitness = None
        self._layer = None
        self.heterogeneous = (
            np.isscalar(value) if heterogeneous is None else heterogeneous
        )
        self.value = value

    def __repr__(self) -> str:
        params = f"{self.value:.2f}"
        if self.heterogeneous:
            params += f", heterogeneous={self.heterogeneous}"
        return f"{self.__class__.__name__}({params})"

    def __copy__(self) -> Self:
        obj = self.__class__(self.value, heterogeneous=self.heterogeneous)
        if self._fitness is not None:
            obj._fitness = self._fitness.copy()
        obj.layer = self.layer
        return obj

    @property
    def dtype(self) -> np.dtype:
        """The data type of the parameter values."""
        return self.value.dtype

    @property
    def value(self) -> np.number:
        """Parameter value."""
        return self._value

    @value.setter
    def value(self, value: float | np.ndarray) -> None:
        self.check_value(value)
        if np.isscalar(value):
            value = float(value)
            self._value = np.dtype(type(value)).type(value)
        else:
            value = np.asarray(value)
            if value.ndim != 1:
                errmsg = "parameter values must be a scalar or 1D array"
                raise ValueError(errmsg)
            if self._layer is not None and len(value) != self.model.n_nodes:
                errmsg = "parameter values length does not match the number of nodes"
                raise ValueError(errmsg)
            self.value = value.mean()
            self._fitness = value - self.value / 2
            self.heterogeneous = True

    @property
    def values(self) -> np.ndarray:
        """Node-specific parameters allowing for heterogeneity."""
        if self.heterogeneous:
            if self._fitness is None:
                errmsg = "node fitnesses are not initialized"
                raise AttributeError(errmsg)
            return (self.value / 2 + self._fitness).astype(self.dtype)
        return np.full(self.model.n_nodes, self.value / 2, dtype=self.dtype)

    @property
    def model(self) -> "GRGG":
        """The parent :class:`grgg.GRGG` instance."""
        return self.layer.model

    @property
    def layer(self) -> "AbstractGRGGLayer":
        """The parent GRGG layer."""
        if self._layer is None:
            errmsg = "parameter is not attached to a GRGG layer"
            raise AttributeError(errmsg)
        layer = self._layer()
        if layer is None:
            errmsg = "the parent GRGG layer has been deleted"
            raise ReferenceError(errmsg)
        return layer

    @layer.setter
    def layer(self, layer: "AbstractGRGGLayer") -> None:
        self._layer = weakref.ref(layer)
        if self.heterogeneous:
            if self._fitness is None:
                self._fitness = np.zeros(layer.model.n_nodes)
            elif self.model.n_nodes != len(self._fitness):
                errmsg = "fitness array length does not match the number of nodes"
                raise ValueError(errmsg)
        else:
            self._fitness = None

    @property
    def heterogeneous(self) -> bool:
        """Whether the parameter is heterogeneous (varies across nodes)."""
        return self._heterogeneous

    @heterogeneous.setter
    def heterogeneous(self, heterogeneous: bool) -> None:
        self._heterogeneous = bool(heterogeneous)
        if self.heterogeneous and self._layer is not None:
            self.layer = self.layer

    @property
    def outer(self) -> Outer:
        """Outer sum of parameter values for node pairs."""
        return Outer(self.values, op=np.add)

    def update(
        self,
        value: float | np.ndarray | None,
        *,
        heterogeneous: bool | None = None,
    ) -> None:
        """Update parameter value and heterogeneity.

        Parameters
        ----------
        value
            New parameter value. If `None`, the current value is kept.
        heterogeneous
            Whether the parameter is heterogeneous (varies across nodes).
            If `None`, the current setting is kept.
        """
        if value is not None:
            self.value = value
        if heterogeneous is not None:
            self.heterogeneous = heterogeneous

    def check_value(self, value: float | np.ndarray) -> None:
        """Check if the parameter value is valid."""


class Beta(CouplingParameter):
    """Beta parameter (inverse temperature).

    It controls the strength of the coupling between the network topology
    and the underlying geometry.

    Attributes
    ----------
    value
        Parameter value.
    heterogeneous
        Whether the parameter is heterogeneous (varies across nodes).
    """

    def __init__(self, value: float | None = None, *args: Any, **kwargs: Any) -> None:
        if value is None:
            value = options.layer.beta
        super().__init__(value, *args, **kwargs)

    def check_value(self, value: float | np.ndarray) -> None:
        if np.any(value < 0):
            errmsg = "beta must be non-negative"
            raise ValueError(errmsg)


class Mu(CouplingParameter):
    """Mu parameter (chemical potential).

    It controls the overall density of the network.

    Attributes
    ----------
    value
        Parameter value.
    heterogeneous
        Whether the parameter is heterogeneous (varies across nodes).
    """

    def __init__(self, value: float | None = None, *args: Any, **kwargs: Any) -> None:
        if value is None:
            value = options.layer.mu
        super().__init__(value, *args, **kwargs)
