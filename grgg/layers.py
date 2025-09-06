import weakref
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Self

import numpy as np
from scipy.special import expit

from . import options
from .manifolds import CompactManifold
from .parameters import Beta, Mu

if TYPE_CHECKING:
    from . import GRGG


__all__ = ("Similarity", "Complementarity")


class AbstractGRGGLayer(ABC):
    """Abstract base class for GRGG layers."""

    def __init__(
        self,
        beta: float | Beta | None = None,
        mu: float | Mu | None = None,
        *,
        log: bool | None = None,
        eps: float | None = None,
    ) -> None:
        if beta is None:
            beta = Beta(options.layer.beta)
        elif not isinstance(beta, Beta):
            beta = Beta(beta)
        if mu is None:
            mu = Mu(options.layer.mu)
        elif not isinstance(mu, Mu):
            mu = Mu(mu)
        if log is None:
            log = options.layer.log
        if eps is None:
            eps = options.layer.eps
        self.beta = beta
        self.mu = mu
        self.log = bool(log)
        self.eps = float(eps)
        self._model = None

    def __repr__(self) -> str:
        cn = self.__class__.__name__
        return f"{cn}({self.beta}, {self.mu}, log={self.log})"

    def __copy__(self) -> Self:
        return self.__class__(self.beta.copy(), self.mu.copy())

    def __call__(self, g: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
        coupling = self.coupling(g, *args, **kwargs)
        return expit(-coupling)

    @property
    def manifold(self) -> CompactManifold:
        """The parent model's manifold."""
        return self.model.manifold

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
        from grgg import GRGG

        if not isinstance(model, GRGG):
            errmsg = "'model' must be a 'GRGG' instance"
            raise TypeError(errmsg)
        self._model = weakref.ref(model)

    @property
    def max_energy(self) -> float:
        return float(self.energy(self.manifold.diameter))

    def copy(self) -> Self:
        return self.__copy__()

    @abstractmethod
    def energy(self, g: np.ndarray) -> np.ndarray:
        """Evaluate the energy function for geodesic distances `g`."""

    def coupling(self, g: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
        """Evaluate the coupling function for geodesic distances `g`."""
        d = self.manifold.dim
        beta, mu = self._get_params(*args, **kwargs)
        energy = self._energy(g)
        coupling = beta * d * (energy - mu) + np.exp(-beta) * mu * (beta + 1)
        return coupling

    def _get_params(
        self,
        idx: np.ndarray | tuple[np.ndarray, np.ndarray] | None = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get coupling function parameters, handling heterogeneity where necessary."""
        beta = (
            self.beta.outer(idx, **kwargs)
            if self.beta.heterogeneous
            else self.beta.value
        )
        mu = self.mu.outer(idx, **kwargs) if self.mu.heterogeneous else self.mu.value
        return beta, mu

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
