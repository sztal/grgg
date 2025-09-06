import math
from typing import Self

import numpy as np
from scipy.spatial.distance import cdist, pdist

from .manifold import CompactManifold

__all__ = ("Sphere",)


class Sphere(CompactManifold):
    """Class representing a sphere manifold.

    Attributes
    ----------
    dim
        Surface dimension of the sphere.
    r
        Radius of the sphere, by default 1.0.
    """

    def __init__(self, dim: int, r: float = 1.0) -> None:
        super().__init__(dim)
        self.r = float(r)

    @property
    def params(self) -> dict[str, float]:
        """Return the parameters of the sphere."""
        return {**super().params, "r": self.r}

    @property
    def radius(self) -> float:
        """Return the radius of the sphere, alias for :attr:`r`."""
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

    def _pdist(self, X: np.ndarray) -> np.ndarray:
        """Compute pairwise distances between points on the sphere.

        Examples
        --------
        >>> sphere = Sphere(2, 3)
        >>> X = sphere.sample_points(5, random_state=42)
        >>> d = sphere.pdist(X, full=False)
        >>> all(d >= 0) and all(d <= sphere.diameter)
        True
        >>> d.shape
        (10,)
        >>> d = sphere.pdist(X, full=True)
        >>> d.shape
        (5, 5)
        """
        return np.arccos(1 - pdist(X / self.r, metric="cosine")) * self.r

    def _cdist(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute pairwise distances between two sets of points on the sphere.

        Examples
        --------
        >>> sphere = Sphere(2, 3)
        >>> X = sphere.sample_points(5, random_state=42)
        >>> Y = sphere.sample_points(3, random_state=43)
        >>> d = sphere.cdist(X, Y)
        >>> d.shape
        (5, 3)
        >>> bool(np.all(d >= 0) and np.all(d <= sphere.diameter))
        True
        """
        return np.arccos(1 - cdist(X / self.r, Y / self.r, metric="cosine")) * self.r

    def _sample_points(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Sample points uniformly on the surface sphere.

        Examples
        --------
        >>> sphere = Sphere(2, 3)
        >>> points = sphere.sample_points(5, random_state=42)
        >>> points.shape
        (5, 3)
        >>> bool(np.allclose(np.linalg.norm(points, axis=1), sphere.radius))
        True
        """
        points = rng.normal(size=(n, self.embedding_dim))
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        if self.r != 1.0:
            points *= self.r
        return points

    def with_radius(self, r: float) -> Self:
        """Return a copy of the sphere with the given radius.

        Examples
        --------
        >>> sphere = Sphere(2).with_radius(3)
        >>> sphere.r
        3.0
        """
        return self.__class__(self.dim, r)

    def with_volume(self, volume: float) -> Self:
        """Return a copy of the sphere with the given volume.

        Examples
        --------
        >>> sphere = Sphere(2).with_volume(20)
        >>> sphere.volume
        20.0
        """
        d = self.embedding_dim
        r = (volume * math.gamma(d / 2) / (2 * math.pi ** (d / 2))) ** (1 / (d - 1))
        return self.__class__(self.dim, r)
