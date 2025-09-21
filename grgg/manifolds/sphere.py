import math
from typing import Any, Self

import equinox as eqx
import jax.numpy as jnp
from jax.scipy.special import gamma

from grgg._typing import Matrix, Scalar, Vector
from grgg.random import RandomGenerator

from .manifold import CompactManifold


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
    >>> sphere.volume
    113.09733552923255
    >>> sphere.diameter
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

    r: float = eqx.field(static=True)

    def __init__(self, dim: int, r: float = 1.0, **kwargs: Any) -> None:
        super().__init__(dim=dim, **kwargs)
        self.r = float(r)

    def __check_init__(self) -> None:
        if self.r <= 0:
            errmsg = "'r' must be a positive number"
            raise ValueError(errmsg)

    def __copy__(self) -> Self:
        return type(self)(self.dim, self.r)

    def _repr_params(self) -> str:
        return f"r={self.r:.2f}"

    @property
    def radius(self) -> float:
        """Radius of the sphere, alias for :attr:`r`."""
        return self.r

    @property
    def linear_size(self) -> float:
        """Linear size of the sphere."""
        return self.r

    @property
    def volume(self) -> float:
        d = self.embedding_dim
        V = 2 * math.pi ** (d / 2) / math.gamma(d / 2) * self.r ** (d - 1)
        return float(V)

    @property
    def diameter(self) -> float:
        """Maximum distance between two points on the sphere."""
        return math.pi * self.r

    def with_volume(self, volume: float) -> Self:
        """Return a copy of the sphere with the specified volume.

        Examples
        --------
        >>> sphere = Sphere(2).with_volume(20)
        >>> sphere.volume
        20.0
        """
        d = self.embedding_dim
        r = (volume * math.gamma(d / 2) / (2 * math.pi ** (d / 2))) ** (1 / (d - 1))
        return type(self)(self.dim, r)

    def equals(self, other: object) -> bool:
        return super().equals(other) and self.r == other.r

    def metric(self, x: Vector, y: Vector) -> Scalar:
        """Geodesic distance between points on the sphere."""
        if len(x) != self.embedding_dim or len(y) != self.embedding_dim:
            errmsg = "points have incompatible number of coordinates"
            raise ValueError(errmsg)
        x /= self.r
        y /= self.r
        cosine = jnp.clip(jnp.dot(x, y), -1.0, 1.0)
        return jnp.arccos(cosine) * self.r

    def _sample_points(self, n: int, rng: RandomGenerator) -> Matrix:
        """Implementation of point sampling."""
        points = rng.normal((n, self.embedding_dim))
        points /= jnp.linalg.norm(points, axis=1, keepdims=True)
        if self.r != 1.0:
            points *= self.r
        return points

    def distance_density(self, g: jnp.ndarray) -> jnp.ndarray:
        theta = g / self.r
        d = self.dim
        num = gamma((d + 1) / 2) * jnp.sin(theta) ** (d - 1)
        den = jnp.sqrt(math.pi) * gamma(d / 2)
        return num / den

    def cosine_law(
        self, theta: jnp.ndarray, g_ij: jnp.ndarray, g_ik: jnp.ndarray
    ) -> jnp.ndarray:
        """Spherical law of cosines.

        Parameters
        ----------
        theta
            The angle at the focal point `i` between the two points `j` and `k`.
        g_ij
            The geodesic distance between points i and j.
        g_ik
            The geodesic distance between points i and k.

        Returns
        -------
        g_jk
            The geodesic distance between points j and k.
        """
        theta_ij = g_ij / self.r
        theta_ik = g_ik / self.r
        cos_theta_jk = jnp.clip(
            (
                jnp.cos(theta_ij) * jnp.cos(theta_ik)
                + jnp.sin(theta_ij) * jnp.sin(theta_ik) * jnp.cos(theta)
            ),
            -1.0,
            1.0,
        )
        theta_jk = jnp.arccos(cos_theta_jk)
        return theta_jk * self.r
