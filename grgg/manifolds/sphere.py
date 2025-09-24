import math
from typing import Any, ClassVar, Self

import jax.numpy as jnp
from jax.lax import stop_gradient

from grgg._typing import Matrix, Scalar, Vector
from grgg.random import RandomGenerator
from grgg.utils import format_array

from .manifold import (
    CompactManifold,
    ManifoldCosineLawFunction,
    ManifoldDistanceDensityFunction,
    ManifoldMetricFunction,
    ManifoldVolumeFunction,
)

__all__ = ("Sphere",)


class Sphere(CompactManifold):
    """Sphere manifold.

    Attributes
    ----------
    dim
        Surface dimension of the sphere.
    r
        Radius of the sphere.

    Examples
    --------
    >>> sphere = Sphere(2, r=3.0)
    >>> sphere
    Sphere(2, r=3.00)
    >>> sphere.volume.item()
    113.09733552923255
    >>> sphere.diameter.item()
    9.42477796076938
    >>> points = sphere.sample_points(10, rng=42)
    >>> points.shape
    (10, 3)
    >>> bool(jnp.allclose(jnp.linalg.norm(points, axis=1), sphere.r))
    True

    Compare distances with the reference implementation from :mod:`scipy`.
    >>> from scipy.spatial.distance import pdist
    >>> r = sphere.r
    >>> dists = sphere.distances(points)
    >>> ref_dists = jnp.arccos(1 - pdist(points / r, metric="cosine")) * r
    >>> bool(jnp.allclose(dists, ref_dists))
    True
    """

    r: Scalar

    def __init__(self, dim: int, r: Scalar = 1.0, **kwargs: Any) -> None:
        super().__init__(dim=dim, **kwargs)
        r = jnp.asarray(r, dtype=float)
        self.r = stop_gradient(r)

    def __check_init__(self) -> None:
        if not jnp.isscalar(self.r):
            errmsg = "radius must be a scalar"
            raise ValueError(errmsg)

    #     if self.r < 0:
    #         errmsg = "radius cannot be negative"
    #         raise ValueError(errmsg)

    def __copy__(self) -> Self:
        return type(self)(self.dim, self.r)

    def _repr_params(self) -> str:
        if self.r.size == 1:
            return f"r={self.r:.2f}"
        return f"r={format_array(self.r)}"

    @property
    def radius(self) -> float:
        """Radius of the sphere, alias for :attr:`r`."""
        return self.r

    @property
    def linear_size(self) -> float:
        """Linear size of the sphere."""
        return self.r

    @property
    def volume(self) -> Scalar:
        """Surface volume of the sphere."""
        return self.compute.volume(self.dim, self.r)

    @property
    def diameter(self) -> float:
        """Maximum distance between two points on the sphere."""
        return math.pi * self.r

    def with_volume(self, volume: float) -> Self:
        """Return a copy of the sphere with the specified volume.

        Examples
        --------
        >>> sphere = Sphere(2).with_volume(20)
        >>> sphere.volume.item()
        20.0
        """
        d = self.embedding_dim
        r = (volume * math.gamma(d / 2) / (2 * math.pi ** (d / 2))) ** (1 / (d - 1))
        return type(self)(self.dim, r)

    def equals(self, other: object) -> bool:
        return super().equals(other) and self.r == other.r

    def metric(self, x: Vector, y: Vector) -> Scalar:
        """Geodesic distance between points on the sphere."""
        return self.compute.metric(x, y, self.dim, self.r)

    def _sample_points(self, n: int, rng: RandomGenerator) -> Matrix:
        """Implementation of point sampling."""
        points = rng.normal((n, self.embedding_dim))
        points /= jnp.linalg.norm(points, axis=1, keepdims=True)
        if self.r != 1.0:
            points *= self.r
        return points


class SphereVolume(ManifoldVolumeFunction):
    """Volume function for the sphere manifold."""

    manifold_type: ClassVar[type[Sphere]] = Sphere

    def unit(self, dim: int) -> float:
        """Compute volume of the sphere with unit radius."""
        return 2 * math.pi ** ((dim + 1) / 2) / math.gamma((dim + 1) / 2)

    def kernel(self, dim: int, r: float = 1.0) -> jnp.ndarray:
        return r**dim


class SphereMetric(ManifoldMetricFunction):
    """Metric (geodesic distance) function for the sphere manifold."""

    manifold_type: ClassVar[type[Sphere]] = Sphere

    def unit(self, x: Vector, y: Vector, dim: int) -> jnp.ndarray:
        embedding_dim = dim + 1
        if len(x) != embedding_dim or len(y) != embedding_dim:
            errmsg = "points have incompatible number of coordinates"
            raise ValueError(errmsg)
        cosine = jnp.clip(jnp.dot(x, y), -1.0, 1.0)
        return jnp.arccos(cosine)


class SphereDistanceDensity(ManifoldDistanceDensityFunction):
    """Distance density function for the sphere manifold.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from grgg.manifolds import Sphere
    >>> sphere = Sphere(2, r=1.0)
    >>> g = jnp.linspace(0, sphere.diameter, 1000)
    >>> pd = sphere.distance_density(g)
    >>> jnp.trapezoid(pd, g).item()  # should be close to 1
    1.0

    Check non-unit radius and other dimensionalities.
    >>> sphere = Sphere(1, r=5.0)
    >>> g = jnp.linspace(0, sphere.diameter, 1000)
    >>> pd = sphere.distance_density(g)
    >>> jnp.trapezoid(pd, g).item()  # should be close to 1
    1.0

    >>> sphere = Sphere(3, r=2.0)
    >>> g = jnp.linspace(0, sphere.diameter, 1000)
    >>> pd = sphere.distance_density(g)
    >>> jnp.trapezoid(pd, g).item()  # should be close to 1
    1.0
    """

    manifold_type: ClassVar[type[Sphere]] = Sphere

    def constant(self, dim: int, r: float) -> float:
        """Constant depending only on the manifold."""
        num = math.gamma((dim + 1) / 2)
        den = math.sqrt(math.pi) * math.gamma(dim / 2) * r
        return num / den

    def kernel(self, g: jnp.ndarray, dim: int, r: float) -> jnp.ndarray:
        return jnp.sin(g / r) ** (dim - 1)


class SphereCosineLaw(ManifoldCosineLawFunction):
    """Cosine law function for the sphere manifold.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from grgg.manifolds import Sphere
    >>> sphere = Sphere(2, r=1.0)
    >>> a = 60 / 180 * jnp.pi
    >>> b = 75 / 180 * jnp.pi
    >>> C = 90 / 180 * jnp.pi
    >>> sphere.cosine_law(C, a, b).item() / jnp.pi * 180  # angle in degrees
    82.56453563
    """

    manifold_type: ClassVar[type[Sphere]] = Sphere

    def unit(
        self,
        theta: jnp.ndarray,
        theta_ij: jnp.ndarray,
        theta_ik: jnp.ndarray,
    ) -> jnp.ndarray:
        cos_theta_jk = jnp.clip(
            (
                jnp.cos(theta_ij) * jnp.cos(theta_ik)
                + jnp.sin(theta_ij) * jnp.sin(theta_ik) * jnp.cos(theta)
            ),
            -1.0,
            1.0,
        )
        theta_jk = jnp.arccos(cos_theta_jk)
        return theta_jk
