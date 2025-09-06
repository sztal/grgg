import weakref
from functools import singledispatchmethod
from typing import TYPE_CHECKING, Self

import numpy as np

if TYPE_CHECKING:
    from . import GRGG
    from .layers import AbstractGRGGLayer


class CouplingParameter:
    """Base class for coupling parameters.

    Attributes
    ----------
    value
        Parameter value.
    heterogeneous
        Whether the parameter is heterogeneous (varies across nodes).
    """

    def __init__(self, value: float, *, heterogeneous: bool = False) -> None:
        self.value = float(value)
        self._heterogeneous = heterogeneous
        self._fitness = None
        self._layer = None

    def __repr__(self) -> str:
        params = f"{self.value:.2f}"
        if self.heterogeneous:
            params += f", heterogeneous={self.heterogeneous}"
        return f"{self.__class__.__name__}({params})"

    def __copy__(self) -> Self:
        obj = self.__class__(self.value, hterogeneous=self.heterogeneous)
        if self._fitness is not None:
            obj._fitness = self._fitness.copy()
        obj.layer = self.layer
        return obj

    @property
    def values(self) -> np.ndarray:
        """Node-specific parameters allowing for heterogeneity."""
        if self.heterogeneous:
            if self._fitness is None:
                errmsg = "node fitnesses are not initialized"
                raise AttributeError(errmsg)
            return self.value + self._fitness
        return np.fill(self.model.n_nodes, self.value)

    @property
    def model(self) -> "GRGG":
        """The parent :class:`grgg.GRGG` instance."""
        return self.coupling.model

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
        if not isinstance(layer, AbstractGRGGLayer):
            errmsg = "'layer' must be an 'AbstractGRGGLayer' instance"
            raise TypeError(errmsg)
        self._layer = weakref.ref(layer)
        if (
            self.heterogeneous
            and self._fitness is None
            or self.model.n_nodes != len(self._fitness)
        ):
            self._fitness = np.zeros(layer.model.n_nodes)

    @property
    def heterogeneous(self) -> bool:
        """Whether the parameter is heterogeneous (varies across nodes)."""
        return self._heterogeneous

    @heterogeneous.setter
    def heterogeneous(self, heterogeneous: bool) -> None:
        self._heterogeneous = bool(heterogeneous)
        if self.heterogeneous and self._layer is not None:
            self.layer = self.layer

    @singledispatchmethod
    def outer(
        self, i: np.ndarray | slice | None = None, *, full: bool = False
    ) -> np.ndarray:
        """Compute the outer sum of parameter values for node pairs `(i, j)`.

        Parameters
        ----------
        i
            Node indices.
        full
            Whether to return a full matrix or only the lower triangular part.
        """
        values = self.values[i] if i is not None else self.values
        outer = values[:, None] + values
        if not full:
            outer = outer[np.tril_indices_from(outer, k=-1)]
        return outer

    @outer.register
    def _(self, ij: tuple) -> np.ndarray:
        """Compute the outer sum of parameter values for node pairs `(i, j)`.

        Parameters
        ----------
        ij
            Tuple of node index arrays `(i, j)`.
        """
        i, j = ij
        values = self.values
        return values[i, None] + values[j]


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
