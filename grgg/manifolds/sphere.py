import math
from typing import Self

import equinox as eqx
import jax.numpy as jnp

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

    r: float = eqx.field(static=True, converter=float)

    def __init__(self, dim: int, r: float = 1.0) -> None:
        super().__init__(dim)
        self.r = float(r)

    def __copy__(self) -> Self:
        return type(self)(self.dim, self.r)

    def _repr_params(self) -> str:
        return f"r={self.r:.2f}"

    @property
    def radius(self) -> float:
        """Radius of the sphere, alias for :attr:`r`."""
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
