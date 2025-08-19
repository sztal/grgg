import math
from abc import ABC, abstractmethod
from functools import singledispatchmethod
from typing import Any, Self

import numpy as np

from . import options
from .manifolds import CompactManifold, Manifold

__all__ = ("Similarity", "Complementarity")


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
            logspace = options.kernel.logspace
        if eps is None:
            eps = options.kernel.eps
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
        attrs = [
            f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in self.repr_attrs.items()
        ]
        return f"{cn}({", ".join(attrs)})"

    def __call__(self, d: np.ndarray) -> np.ndarray:
        """Evaluate the kernel function for distances `d`."""
        return self.kernel(d)

    def __copy__(self) -> Self:
        """Create a copy of the kernel instance with updated parameters."""
        kwargs = {
            **self.params,
            "logspace": self.logspace,
            "eps": self.eps,
        }
        return self.__class__(**kwargs)

    @property
    def params(self) -> dict[str, float]:
        return {"mu": self.mu, "beta": self.beta}

    @property
    def repr_attrs(self) -> dict[str, Any]:
        """Return parameters suitable for printing."""
        return {"mu": self.mu, "beta": self.beta, "logspace": self.logspace}

    @abstractmethod
    def kernel(self, r: np.ndarray) -> np.ndarray:
        """Evaluate the kernel function."""
        r = np.maximum(r, self.eps)  # ensure no zero relation scores
        if self.logspace:
            r = np.log(r)
        # This is a trick that allows to safely interpolate
        # between hard RGGs and ER graphs
        if math.isinf(self.beta):
            return np.where(r < self.mu, -np.inf, np.inf)
        alpha = math.exp(-self.beta) * self.mu * (self.beta - 1)
        return alpha + self.beta * (r - self.mu)

    @classmethod
    def from_manifold(cls, manifold, **kwargs: Any) -> Self:
        """Create a kernel instance from a manifold."""
        default_params = cls.params_from_manifold(manifold, **kwargs)
        return cls(**default_params)

    @singledispatchmethod
    @classmethod
    def params_from_manifold(cls, manifold, **kwargs: Any) -> dict[str, Any]:  # noqa
        errmsg = f"not implemented for '{manifold.__class__.__name__}'"
        raise NotImplementedError(errmsg)

    @params_from_manifold.register
    @classmethod
    def _(cls, manifold: Manifold, **kwargs: Any) -> dict[str, Any]:
        """Create a kernel instance from a sphere."""
        if (key := "beta") not in kwargs:
            kwargs[key] = 1.5 * manifold.dim
        if (key := "mu") not in kwargs:
            if np.isinf(kwargs["beta"]):
                mu = manifold.surface_area ** (1 / manifold.embedding_dim) / np.pi
                if kwargs.get("logspace", options.kernel.logspace):
                    mu = np.log(mu)
                kwargs[key] = mu
            else:
                kwargs[key] = 0.0
        return kwargs

    def copy(self) -> Self:
        """Create a copy of the kernel instance with updated parameters."""
        return self.__copy__()


class Similarity(AbstractGeometricKernel):
    """Kernel for the Similarity-RGG model."""

    def kernel(self, d: np.ndarray) -> np.ndarray:
        """Evaluate the similarity kernel function."""
        return super().kernel(d)


class Complementarity(AbstractGeometricKernel):
    """Kernel for the Complementarity-RGG model."""

    def __init__(
        self, mu: float, beta: float, max_distance: float, **kwargs: Any
    ) -> None:
        """Initialize the complementarity kernel."""
        super().__init__(mu, beta, **kwargs)
        self.max_distance = float(max_distance)

    @property
    def params(self) -> dict[str, float]:
        """Return the parameters of the complementarity kernel."""
        params = super().params
        params["max_distance"] = self.max_distance
        return params

    def kernel(self, d: np.ndarray) -> np.ndarray:
        """Evaluate the complementarity kernel function."""
        r = self.max_distance - d
        return super().kernel(r)

    @singledispatchmethod
    def params_from_manifold(cls, manifold, **kwargs: Any) -> dict[str, Any]:
        return super().params_from_manifold(manifold, **kwargs)

    @params_from_manifold.register
    @classmethod
    def _(cls, manifold: Manifold, **kwargs: Any) -> dict[str, float]:
        if not isinstance(manifold, CompactManifold):
            errmsg = "complementarity kernel requires a compact manifold"
            raise TypeError(errmsg)
        params = super().params_from_manifold(manifold, **kwargs)
        if (key := "max_distance") not in params:
            params[key] = manifold.max_distance
        return params
