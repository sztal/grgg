import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any

from scipy.integrate import quad

from .manifolds import CompactManifold, Manifold, Sphere

if TYPE_CHECKING:
    from .model import GRGG


class Integration:
    """Namespace for integration methods for the GRGG model.

    Attributes
    ----------
    kbar
        Integral for computing the average degree :math:`\bar{k}`.
    """

    def __init__(self, model: "GRGG") -> None:
        self.kbar = KBarIntegral(model)


class Integral(ABC):
    """Abstract base class for numerical integration over GRGG ensembles."""

    def __init__(self, model: "GRGG") -> None:
        self.model = model

    def __call__(self, **kwargs: Any) -> tuple[float | Any]:
        return self.integrate(**kwargs)

    def integrate(
        self, a: float | None = None, b: float | None = None, **kwargs: Any
    ) -> tuple[float | Any]:
        """Perform numerical integration over the GRGG model."""
        integrand = self.make_integrand()
        a, b = self.integration_limits(self.model.manifold, a, b)
        kwargs = self.integration_opts(self.model.manifold, **kwargs)
        value, *info = quad(integrand, a, b, **kwargs)
        return value, info

    def make_integrand(self) -> Callable[[float], float]:
        """Make integrand function."""
        return self._make_integrand(self.model.manifold)

    @abstractmethod
    def _make_integrand(self, manifold: Manifold) -> Callable[[float], float]:
        """Create the integrand function based on the manifold."""

    @singledispatchmethod
    def integration_opts(self, manifold: Manifold, **kwargs: Any) -> dict[str, Any]:  # noqa
        """Get integration options based on the manifold."""
        errmsg = f"'{manifold.__class__.__name__}' is not supported."
        raise NotImplementedError(errmsg)

    @integration_opts.register
    def _(self, manifold: CompactManifold, **kwargs: Any) -> dict[str, Any]:
        """Get integration options for a sphere."""
        p = manifold.volume ** (-1 / manifold.embedding_dim)  # heuristic
        return {"points": [p, math.pi - p], **kwargs}

    @singledispatchmethod
    def integration_limits(
        self,
        manifold: Manifold,
        a: float | None = None,  # noqa
        b: float | None = None,  # noqa
    ) -> tuple[float, float]:
        """Get integration limits based on the manifold."""
        errmsg = f"'{manifold.__class__.__name__}' is not supported."
        raise NotImplementedError(errmsg)

    @integration_limits.register(Sphere)
    def _(
        self,
        manifold: Sphere,  # noqa
        a: float | None = None,
        b: float | None = None,
    ) -> tuple[float, float]:
        """Get integration limits for a sphere."""
        return 0.0 if a is None else a, math.pi if b is None else b


class KBarIntegral(Integral):
    """Integral for computing the average degree :math:`\bar{k}` of the GRGG model."""

    @singledispatchmethod
    def _make_integrand(self, manifold: Manifold) -> Callable[[float], float]:
        """Make integrand function for average degree."""
        errmsg = f"'{manifold.__class__.__name__}' is not supported."
        raise NotImplementedError(errmsg)

    @_make_integrand.register(Sphere)
    def _(self, manifold: Sphere) -> Callable[[float], float]:
        """Make integrand function for average degree on a sphere."""
        d = manifold.dim
        R = manifold.radius

        def integrand(theta: float) -> float:
            return math.sin(theta) ** (d - 1) * self.model.dist2prob(R * theta)

        return integrand

    def integrate(self, *args: Any, **kwargs: Any) -> tuple[float, Any]:
        """Perform numerical integration to compute the average degree."""
        n = self.model.n_nodes
        d = self.model.manifold.dim
        R = self.model.manifold.radius
        rho = self.model.rho / n * (n - 1)  # exclude self-loops
        dV = Sphere(d - 1).volume  # volume element
        integral, info = super().integrate(*args, **kwargs)  # type: ignore
        integral *= rho * dV * R**d
        return integral, info
