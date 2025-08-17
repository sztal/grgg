from abc import ABC, abstractmethod
from functools import singledispatchmethod
from typing import Any, Self

import numpy as np
from geomstats.geometry.manifold import Manifold

from . import options
from .manifolds import Sphere


class AbstractGeometricKernel(ABC):
    """Abstract base class for geometric kernels."""

    def __init__(
        self,
        mu: float,
        beta: float,
        *,
        logspace: bool | None = None,
        eps: float | None = None,
    ) -> None:
        if logspace is None:
            logspace = options.logspace
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
        self.logspace = bool(logspace)
        self.eps = eps

    def __repr__(self) -> str:
        cn = self.__class__.__name__
        params = {**self.params, "logspace": self.logspace}
        params_str = ", ".join(f"{k}={v}" for k, v in params.items())
        return f"{cn}({params_str})"

    def __call__(self, d: np.ndarray) -> np.ndarray:
        """Evaluate the kernel function for distances `d`."""
        r = self.relation_scores(d)
        r = np.maximum(r, self.eps)  # ensure no zero relation scores
        if self.logspace:
            r = np.log(r)
        return self.kernel(r)

    @property
    def params(self) -> dict[str, float]:
        return {"mu": self.mu, "beta": self.beta}

    @abstractmethod
    def relation_scores(self, d: np.ndarray) -> np.ndarray:
        """Convert manifold distances to relation scores."""

    def kernel(self, d: np.ndarray) -> np.ndarray:
        """Evaluate the kernel function."""
        if np.isinf(self.beta):
            return np.where(d <= self.mu, -np.inf, np.inf)
        return self.beta * d - self.mu

    def __copy__(self) -> Self:
        """Create a copy of the kernel instance with updated parameters."""
        kwargs = {
            **self.params,
            "logspace": self.logspace,
            "eps": self.eps,
        }
        return self.__class__(**kwargs)

    @classmethod
    def from_manifold(cls, manifold: Manifold, n_nodes: int, **kwargs: Any) -> Self:
        """Create a kernel instance from a manifold."""
        default_params = cls.params_from_manifold(manifold, n_nodes, **kwargs)
        return cls(**default_params)

    @singledispatchmethod
    @classmethod
    def params_from_manifold(
        cls,
        manifold: Manifold,
        n_nodes: int,  # noqa
        **kwargs: Any,  # noqa
    ) -> dict[str, Any]:
        errmsg = f"not implemented for '{manifold.__class__.__name__}'"
        raise NotImplementedError(errmsg)

    @params_from_manifold.register
    @classmethod
    def _(cls, manifold: Sphere, n_nodes: int, **kwargs: Any) -> dict[str, float]:
        return cls.params_from_sphere(manifold, n_nodes, **kwargs)

    @classmethod
    def params_from_sphere(
        cls, sphere: Sphere, n_nodes: int, **kwargs: Any
    ) -> dict[str, Any]:
        """Create a kernel instance from a sphere."""
        if (key := "mu") not in kwargs:
            kwargs[key] = sphere.radius(n_nodes) * np.pi / 2
        if (key := "beta") not in kwargs:
            kwargs[key] = 1.5 * sphere.dim
        return kwargs

    def copy(self) -> Self:
        """Create a copy of the kernel instance with updated parameters."""
        return self.__copy__()


class Similarity(AbstractGeometricKernel):
    """Kernel for the Similarity-RGG model."""

    def relation_scores(self, d: np.ndarray) -> np.ndarray:
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

    def relation_scores(self, d: np.ndarray) -> np.ndarray:
        return self.dmax - d

    @classmethod
    def params_from_sphere(
        cls, sphere: Sphere, n_nodes: int, **kwargs: Any
    ) -> dict[str, float]:
        params = super().params_from_sphere(sphere, n_nodes, **kwargs)
        if (key := "dmax") not in params:
            params[key] = sphere.radius(n_nodes) * np.pi
        return params
