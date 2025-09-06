import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.integrate import quad

from .manifolds import Manifold, Sphere
from .utils import batch_arrays

if TYPE_CHECKING:
    from .model import GRGG

NumericT = np.ndarray | np.number
IntegrandT = Callable[..., NumericT]
IntOutT = tuple[NumericT, dict[str, Any]]


class Integration:
    """Namespace for integration methods for the GRGG model.

    Attributes
    ----------
    degree
        Integral for computing the average degree :math:`\bar{k}`.
    """

    def __init__(self, model: "GRGG") -> None:
        self.degree = DegreeIntegral(model)


class Integral(ABC):
    """Abstract base class for numerical integration over GRGG ensembles."""

    def __init__(self, model: "GRGG") -> None:
        self.model = model

    def __call__(self, *args: Any, **kwargs: Any) -> IntOutT:
        """Perform numerical integration over the GRGG model."""
        integrand, a, b, kwargs = self.prepare(**kwargs)
        integral, info = self.integrate(integrand, a, b, *args)
        integral = self.postprocess(integral)
        return integral, info

    def integrate(
        self, integrand: IntegrandT, a: float, b: float, *args: Any
    ) -> IntOutT:
        """Perform numerical integration."""
        integral, info = quad(integrand, a, b, args)
        return integral, info

    @abstractmethod
    def prepare(
        self, a: float | None = None, b: float | None = None, **kwargs: Any
    ) -> tuple[IntegrandT, float, float, dict[str, Any]]:
        """Prepare the integrand and integration limits."""

    @abstractmethod
    def postprocess(self, integral: NumericT) -> NumericT:
        """Postprocess the integral result."""


class DegreeIntegral(Integral):
    """Integral for computing expected node degree(s).

    Attributes
    ----------
    model
        A GRGG model instance.
    """

    @property
    def scale(self) -> float:
        """Scaling factor for the integral."""
        delta = self.model.delta
        d = self.model.manifold.dim
        R = self.model.manifold.radius
        dV = Sphere(d - 1).volume
        return delta * dV * R**d

    def prepare(
        self, a: float | None = None, b: float | None = None, **kwargs: Any
    ) -> tuple[IntegrandT, float, float, dict[str, Any]]:
        """Prepare the integrand and integration limits."""
        if self.model.is_heterogeneous:
            integrand = self._heterogeneous_integrand(self.model.manifold, **kwargs)
        else:
            integrand = self._homogeneous_integrand(self.model.manifold, **kwargs)
        a = 0.0 if a is None else a
        b = math.pi if b is None else b
        # Heuristic for finding important points to evaluate during integration
        p = self.model.manifold.volume ** (-1 / self.model.manifold.embedding_dim)
        # points that have to be sampled during integration
        # this helps the integration find areas with high probability
        kwargs = {
            "points": [p, math.pi - p],
            **kwargs,
        }
        return integrand, a, b, kwargs

    def postprocess(self, integral: NumericT) -> NumericT:
        """Postprocess the integral result."""
        if self.model.is_heterogeneous:
            C = self.scale / self.model.n_nodes
        else:
            C = (1 - 1 / self.model.n_nodes) * self.scale  # exclude self-loops
        return C * integral

    @singledispatchmethod
    def _homogeneous_integrand(self, manifold: Manifold) -> Callable[[float], float]:
        """Make integrand function for a homogeneous GRGG model."""
        errmsg = f"'{manifold.__class__.__name__}' is not supported."
        raise NotImplementedError(errmsg)

    @_homogeneous_integrand.register
    def _(self, manifold: Sphere) -> Callable[[float], float]:
        """Make integrand function for a sphere."""
        d = manifold.dim
        R = manifold.radius

        def integrand(theta: float) -> float:
            return math.sin(theta) ** (d - 1) * self.model.dist2prob(theta * R)

        return integrand

    @singledispatchmethod
    def _heterogeneous_integrand(
        self,
        manifold: Manifold,
        **kwargs: Any,  # noqa
    ) -> Callable[[float, int], float]:
        """Make integrand function for a heterogeneous GRGG model."""
        errmsg = f"'{manifold.__class__.__name__}' is not supported."
        raise NotImplementedError(errmsg)

    @_heterogeneous_integrand.register
    def _(self, manifold: Sphere, **kwargs: Any) -> Callable[[float, int], float]:
        d = manifold.dim
        R = manifold.radius

        def integrand(theta: float, i: int) -> float:
            g = R * theta
            return math.sin(theta) ** (d - 1) * self.inner_sum(g, i, **kwargs)

        return integrand

    def inner_sum(self, g: NumericT, i: int, **kwargs: Any) -> float:
        """Iterate over parameter values.

        Parameters
        ----------
        g
            Geodesic distances.
        **kwargs
            Passeed to :func:`~grgg.utils.batch_arrays`.
        """
        return self._inner_sum(g, i, **kwargs)

    def _inner_sum(self, g: NumericT, i: int, batch_size: int | None = None) -> float:
        batch_size = batch_size if batch_size is not None else 1000
        if batch_size <= 0:
            batch_size = self.model.n_nodes
        out = 0.0
        indices = np.arange(self.model.n_nodes)
        indices = indices[indices != i]
        for j in batch_arrays(indices, batch_size=batch_size):
            out += self.model.dist2prob(g, i, *j).sum()
        return out
