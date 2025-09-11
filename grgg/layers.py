import weakref
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Self

import numpy as np
from scipy.special import expit

from grgg.manifolds import CompactManifold
from grgg.parameters import Beta, Mu

from . import options

if TYPE_CHECKING:
    from .model import GRGG


__all__ = ("Similarity", "Complementarity")


class AbstractGRGGLayer(ABC):
    """Abstract base class for GRGG layers.

    Examples
    --------
    Check coupling broadcasting in the homogeneous case.
    >>> from grgg import GRGG, Similarity
    >>> model = GRGG(100, 2, Similarity())
    >>> L = model.layers[0]
    >>> float(L.coupling(1))
    0.0
    >>> float(L.coupling(1, [1, 2], [3, 5, 7]))
    0.0
    >>> L.coupling([1, 1], [1, 2], [3, 5, 7])
    array([0., 0.])

    Check coupling broadcasting in the heterogeneous case.
    >>> model = GRGG(100, 2, Similarity(Mu([0]*100), log=True))
    >>> L = model.layers[0]
    >>> float(L.coupling(1, 1, 3))
    0.0
    >>> L.coupling(1, [1, 2], [3, 5])
    array([[0., 0.],
           [0., 0.]])
    >>> L.coupling(1, 3, [1, 2])
    array([0., 0.])
    >>> L.coupling(1, [1, 2], 3)
    array([[0.],
           [0.]])
    >>> L.coupling([1, 1], [[1, 2], [3, 4]])
    array([0., 0.])
    """

    def __init__(
        self,
        beta: float | Beta | None = None,
        mu: float | Mu | None = None,
        *,
        log: bool | None = None,
        eps: float | None = None,
    ) -> None:
        if isinstance(beta, Mu) or isinstance(mu, Beta):
            beta, mu = mu, beta  # type: ignore
        self._model = None
        self._mu = None
        self._beta = None
        self.beta = beta if beta is not None else Beta()
        self.mu = mu if mu is not None else Mu()
        self.log = bool(log if log is not None else options.layer.log)
        self.eps = float(eps if eps is not None else options.layer.eps)

    def __repr__(self) -> str:
        cn = self.__class__.__name__
        return f"{cn}({self.beta}, {self.mu}, log={self.log})"

    def __copy__(self) -> Self:
        return self.__class__(self.beta.copy(), self.mu.copy())

    def __call__(self, coupling: np.ndarray) -> np.ndarray:
        return expit(-coupling)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AbstractGRGGLayer):
            return NotImplemented
        return (
            self.beta == other.beta
            and self.mu == other.mu
            and self.log == other.log
            and self.eps == other.eps
        )

    def __hash__(self) -> int:
        return hash((self.beta, self.mu, self.log, self.eps))

    @property
    def manifold(self) -> CompactManifold:
        """The parent model's manifold."""
        return self.model.manifold

    @property
    def is_heterogeneous(self) -> bool:
        """Whether the layer is heterogeneous."""
        return self.beta.heterogeneous or self.mu.heterogeneous

    @property
    def beta(self) -> Beta:
        """Coupling parameter :math:`\\beta`."""
        return self._beta

    @beta.setter
    def beta(self, beta: Beta | float) -> None:
        if not isinstance(beta, Beta):
            beta = Beta(beta)
        self._beta = beta
        if self._model is not None:
            self.beta.layer = self

    @property
    def mu(self) -> Mu:
        """Coupling parameter :math:`\\mu`."""
        return self._mu

    @mu.setter
    def mu(self, mu: Mu | float) -> None:
        if not isinstance(mu, Mu):
            mu = Mu(mu)
        self._mu = mu
        if self._model is not None:
            self.mu.layer = self

    @property
    def model(self) -> "GRGG":
        """The parent GRGG model."""
        if self._model is None:
            errmsg = "layer is not attached to a GRGG model"
            raise AttributeError(errmsg)
        model = self._model()
        if model is None:
            errmsg = "the parent GRGG model has been deleted"
            raise ReferenceError(errmsg)
        return model

    @model.setter
    def model(self, model: "GRGG") -> None:
        self._model = weakref.ref(model)
        self.beta.layer = self
        self.mu.layer = self

    def copy(self) -> Self:
        return self.__copy__()

    def coupling(self, g: np.ndarray, *idx: slice | np.ndarray) -> np.ndarray:
        """Evaluate the coupling function for geodesic distances `g`."""
        d = self.manifold.dim
        idx = self.model.make_idx(g, *idx)  # type: ignore
        energy = self._energy(g)
        beta = self.beta.outer[idx] if self.beta.heterogeneous else self.beta.value
        mu = self.mu.outer[idx] if self.mu.heterogeneous else self.mu.value
        coupling = beta * d * (energy - mu) + np.exp(-beta) * mu * (beta + 1)
        return coupling

    @abstractmethod
    def energy(self, g: np.ndarray) -> np.ndarray:
        """Evaluate the energy function for geodesic distances `g`."""

    def _energy(self, g: np.ndarray) -> np.ndarray:
        """Evaluate the energy function for geodesic distances `g`."""
        g = np.asarray(g)
        energy = np.maximum(self.energy(g), self.eps)
        if self.log:
            energy = np.log(energy)
        return energy


class Similarity(AbstractGRGGLayer):
    """Similarity GRGG layer."""

    def energy(self, g: np.ndarray) -> np.ndarray:
        """Evaluate the energy function for geodesic distances `g`."""
        return g


class Complementarity(AbstractGRGGLayer):
    """Complementarity GRGG layer."""

    def energy(self, g: np.ndarray) -> np.ndarray:
        """Evaluate the energy function for geodesic distances `g`."""
        return self.manifold.diameter - g
