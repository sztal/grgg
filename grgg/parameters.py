import weakref
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import singledispatchmethod
from typing import TYPE_CHECKING, Self

import numpy as np

from . import options

if TYPE_CHECKING:
    from . import GRGG
    from .layers import AbstractGRGGLayer


IndexT = int | slice | np.ndarray


__all__ = ("Beta", "Mu")


class CouplingParameter(ABC):
    """Base class for coupling parameters.

    Attributes
    ----------
    value
        Parameter value.
    heterogeneous
        Whether the parameter is heterogeneous (varies across nodes).
    """

    def __init__(self, value: float | np.ndarray | None = None) -> None:
        self._value = None
        self._fitness = None
        self._layer = None
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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CouplingParameter):
            return NotImplemented
        if self.heterogeneous != other.heterogeneous:
            return False
        if self.heterogeneous:
            return np.array_equal(self.values, other.values)
        return self.value == other.value

    def __hash__(self) -> int:
        return hash((self.value, self._fitness))

    @property
    @abstractmethod
    def default_value(self) -> float | np.ndarray:
        """The default parameter value."""

    @property
    def heterogeneous(self) -> bool:
        """Whether the parameter is heterogeneous (varies across nodes)."""
        return self._fitness is not None

    @property
    def value(self) -> np.number:
        """Parameter value."""
        return self._value

    @value.setter
    def value(self, value: float | np.ndarray | None) -> None:
        value = self.check_value(value)
        if np.isscalar(value):
            self._value = value
        else:
            self._value = value.mean()
            self._fitness = value - self._value / 2

    @property
    def values(self) -> np.ndarray:
        """Node-specific parameters allowing for heterogeneity."""
        if self.heterogeneous:
            return self.value / 2 + self._fitness
        return np.full(self.model.n_nodes, self.value / 2)

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
    def outer(self) -> "Outer":
        """Outer sum of parameter values for node pairs."""
        return Outer(self.values, op=np.add)

    @abstractmethod
    def check_value(self, value: float | np.ndarray | None) -> float | np.ndarray:
        """Check if the parameter value is valid."""
        if value is None:
            value = self.default_value
        if np.isscalar(value):
            value = float(value)
            return np.dtype(type(value)).type(value)
        value = np.asarray(value)
        if value.ndim != 1:
            errmsg = "parameter values must be a scalar or 1D array"
            raise ValueError(errmsg)
        if self._layer is not None and len(value) != self.model.n_nodes:
            errmsg = "parameter values length does not match the number of nodes"
            raise ValueError(errmsg)
        if not np.isreal(value).all():
            errmsg = "parameter values must be real"
            raise ValueError(errmsg)
        return value

    def copy(self) -> Self:
        """Create a copy of the parameter."""
        return self.__copy__()


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

    Examples
    --------
    >>> Beta()  # default value
    Beta(1.50)
    >>> Beta(2.0)  # homogeneous value
    Beta(2.00)
    >>> Beta([1,2,3])  # heterogeneous value
    Beta(2.00, heterogeneous=True)
    """

    @property
    def default_value(self) -> float:
        return options.layer.beta

    def check_value(self, value: float | np.ndarray) -> float | np.ndarray:
        value = super().check_value(value)
        if np.any(value < 0):
            errmsg = "beta must be non-negative"
            raise ValueError(errmsg)
        return value


class Mu(CouplingParameter):
    """Mu parameter (chemical potential).

    It controls the overall density of the network.

    Attributes
    ----------
    value
        Parameter value.
    heterogeneous
        Whether the parameter is heterogeneous (varies across nodes).

    Examples
    --------
    >>> Mu()  # default value
    Mu(0.00)
    >>> Mu(-1.0)  # homogeneous value
    Mu(-1.00)
    >>> Mu([-1,0,1])  # heterogeneous value
    Mu(0.00, heterogeneous=True)
    """

    @property
    def default_value(self) -> float:
        return options.layer.mu

    def check_value(self, value: float | np.ndarray) -> float | np.ndarray:
        return super().check_value(value)


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

    Examples
    --------
    >>> import numpy as np
    >>> v = np.array([1, 2, 3])
    >>> outer = Outer(v)
    >>> float(outer[0, 1])  # op(v[0], v[1])
    2.0
    >>> float(outer[1, 2])  # op(v[1], v[2])
    6.0
    >>> outer[0]  # op(v, v) lower triangle without diagonal
    Traceback (most recent call last):
    ...
    IndexError: outer requires two indexers
    >>> outer[[0, 2]]
    array([3])
    >>> outer[[[0, 1], [1, 2]]]
    array([2, 6])
    >>> outer[:, :1]
    array([[1],
           [2],
           [3]])
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
        errmsg = "outer requires two indexers"
        raise IndexError(errmsg)

    @__getitem__.register
    def _(self, i: np.ndarray) -> np.ndarray:
        if i.ndim == 1:
            out = self[i, i]
            if not np.isscalar(out):
                out = out[np.tril_indices_from(out, k=-1)]  # type: ignore
            return out
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
            errmsg = "wrong number of indices"
            raise IndexError(errmsg) from None
        vi = self.v1[i]
        vj = self.v2[j]
        if not np.isscalar(vi):
            vi = vi[:, None]
        out = self.op(vi, vj)
        return out
