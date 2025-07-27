from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Self

import numpy as np

from .manifolds import Sphere
from .utils import copy_with_update


@dataclass
class AbstractGeometricKernel(ABC):
    """Abstract base class for geometric kernels."""

    mu: float
    beta: float

    def __post_init__(self) -> None:
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray) and v.size == 1:
                setattr(self, k, v.item())

    @abstractmethod
    def __call__(self, d: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
        """Evaluate the kernel function for distances `d`."""

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
        beta: float | None = None,
    ) -> dict[str, float]:
        if mu is None:
            mu = np.pi * sphere.R / 2
        if beta is None:
            beta = sphere.k * 3 / 2
        return {"mu": mu, "beta": beta}

    def copy(self, **kwargs: Any) -> Self:
        """Create a copy of the kernel instance with updated parameters."""
        return copy_with_update(self, **kwargs)


@dataclass
class Similarity(AbstractGeometricKernel):
    """Kernel for the Similarity-RGG model."""

    def __call__(self, d: np.ndarray) -> np.ndarray:
        """Evaluate the similarity kernel function."""
        return np.exp(self.beta * d - self.mu)


@dataclass
class Complementarity(AbstractGeometricKernel):
    """Kernel for the Complementarity-RGG model."""

    dmax: float

    def __call__(self, d: np.ndarray) -> np.ndarray:
        """Evaluate the complementarity kernel function."""
        return np.exp(self.beta * (self.dmax - d) - self.mu)

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
