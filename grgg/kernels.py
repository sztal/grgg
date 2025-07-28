from abc import ABC, abstractmethod
from typing import Any, Self

import numpy as np

from . import options
from .manifolds import Sphere
from .utils import copy_with_update


class AbstractGeometricKernel(ABC):
    """Abstract base class for geometric kernels."""

    def __init__(
        self,
        mu: float,
        beta: float,
        *,
        logdist: bool | None = None,
        eps: float | None = None,
    ) -> None:
        if logdist is None:
            logdist = options.logdist
        if eps is None:
            eps = options.eps
        eps = float(eps)
        if eps <= 0:
            errmsg = "'eps' must be positive"
            raise ValueError(errmsg)
        if beta < 0:
            errmsg = "'beta' must be non-negative"
            raise ValueError(errmsg)
        self.mu = float(mu)
        self.beta = float(beta)
        self.logdist = bool(logdist)
        self.eps = eps

    def __repr__(self) -> str:
        cn = self.__class__.__name__
        params = {**self.params, "logdist": self.logdist, "eps": self.eps}
        params_str = ", ".join(f"{k}={v}" for k, v in params.items())
        return f"{cn}({params_str})"

    def __call__(self, d: np.ndarray) -> np.ndarray:
        """Evaluate the kernel function for distances `d`."""
        d = self.preprocess(d)
        d = np.maximum(d, self.eps)  # ensure no zero distances
        if self.logdist:
            d = np.log(d)
        return self.kernel(d)

    @property
    def params(self) -> dict[str, float]:
        return {"mu": self.mu, "beta": self.beta}

    @abstractmethod
    def preprocess(self, d: np.ndarray) -> np.ndarray:
        """Process distances before applying the kernel function."""

    def kernel(self, d: np.ndarray) -> np.ndarray:
        """Evaluate the kernel function."""
        if np.isinf(self.beta):
            return self.beta * (d - self.mu)
        return self.beta * d - self.mu

    def __copy__(self) -> Self:
        """Create a copy of the kernel instance with updated parameters."""
        return self.__class__(self.mu, self.beta)

    @classmethod
    def from_sphere(cls, sphere: Sphere, **kwargs: Any) -> Self:
        """Create a kernel instance from a model.

        This allows for setting reasonable defaults for `mu` and `beta`.
        """
        params = cls.params_from_sphere(sphere, **kwargs)
        return cls(**params)

    @classmethod
    def params_from_sphere(
        cls,
        sphere: Sphere,
        *,
        mu: float | None = None,
        beta: float = 1.5,
        logdist: bool | None = None,
        eps: float | None = None,
    ) -> dict[str, float]:
        beta = float(beta * sphere.k)
        if mu is None:
            mu = np.pi * sphere.R / 2
        return {"mu": mu, "beta": beta, "logdist": logdist, "eps": eps}

    def copy(self, **kwargs: Any) -> Self:
        """Create a copy of the kernel instance with updated parameters."""
        return copy_with_update(self, **kwargs)


class Similarity(AbstractGeometricKernel):
    """Kernel for the Similarity-RGG model."""

    def preprocess(self, d: np.ndarray) -> np.ndarray:
        return d


class Complementarity(AbstractGeometricKernel):
    """Kernel for the Complementarity-RGG model."""

    def __init__(self, mu: float, beta: float, dmax: float, **kwargs: Any) -> None:
        """Initialize the complementarity kernel."""
        super().__init__(mu, beta, **kwargs)
        self.dmax = float(dmax)

    @property
    def params(self) -> dict[str, float]:
        """Return the parameters of the complementarity kernel."""
        params = super().params
        params["dmax"] = self.dmax
        return params

    def preprocess(self, d: np.ndarray) -> np.ndarray:
        return self.dmax - d

    def __copy__(self) -> Self:
        """Create a copy of the complementarity kernel instance."""
        return self.__class__(self.mu, self.beta, self.dmax)

    @classmethod
    def params_from_sphere(
        cls, sphere: Sphere, *, dmax: float | None = None, **kwargs: Any
    ) -> dict[str, float]:
        params = super().params_from_sphere(sphere, **kwargs)
        if dmax is None:
            dmax = sphere.R * np.pi
        params["dmax"] = dmax
        return params
