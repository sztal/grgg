import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.integrate import quad

from .manifolds import Manifold, Sphere
from .manifolds.sphere import sphere_surface_area

if TYPE_CHECKING:
    from .model import GRGG


class GRGGIntegration:
    """Namespace for integration methods in GRGG model.

    Attributes
    ----------
    kbar
        Integral for computing the average degree :math:`\bar{k}`.
    """

    def __init__(self, model: "GRGG") -> None:
        self.kbar = KBarIntegral(model)


class GRGGIntegral(ABC):
    """Abstract base class for numerical integration over GRGG model."""

    def __init__(self, model: "GRGG") -> None:
        self.model = model

    def __call__(self) -> float:
        return self.integrate()

    def integrate(self, **kwargs: Any) -> float:
        """Perform numerical integration over the GRGG model."""
        lo = 0.0
        up = getattr(self.model.manifold, "max_distance", np.inf)
        kwargs = {
            "points": [lo, up] if any(k.logspace for k in self.model.kernels) else None,
            "epsabs": 1e-6,
            "epsrel": 1e-6,
            **kwargs,
        }
        integrand = self.make_integrand()
        value, _ = quad(integrand, lo, up, **kwargs)
        return value

    def make_integrand(self) -> Callable[[float], float]:
        """Make integrand function."""
        return self._make_integrand(self.model.manifold)

    @abstractmethod
    def _make_integrand(self, manifold: Manifold) -> Callable[[float], float]:
        """Create the integrand function based on the manifold."""


class KBarIntegral(GRGGIntegral):
    """Integral for computing the average degree :math:`\bar{k}` of the GRGG model."""

    @singledispatchmethod
    def _make_integrand(self, manifold: Manifold) -> Callable[[float], float]:
        """Make integrand function for average degree."""
        errmsg = f"'{manifold.__class__.__name__}' is not supported."
        raise NotImplementedError(errmsg)

    @_make_integrand.register(Sphere)
    def _(self, manifold: Sphere) -> Callable[[float], float]:
        """Make integrand function for average degree on a sphere."""
        R = manifold.radius
        rho = self.model.rho

        def integrand(d: float) -> float:
            r = R * math.sin(d / R)
            S = sphere_surface_area(manifold.dim, r)
            return self.model.edgeprobs(d) * S * rho

        return integrand

    def integrate(self) -> float:
        integral = super().integrate()
        n = self.model.n_nodes
        return integral / n * (n - 1)
