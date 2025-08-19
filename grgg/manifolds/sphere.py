import math
from typing import Self

import numpy as np
from scipy.spatial.distance import cdist, pdist

from .manifold import CompactManifold

__all__ = ("Sphere",)


def sphere_surface_area(k: int, r: float = 1.0) -> float:
    """Surface area of a sphere in `k` embedding dimensions and radius `r`."""
    return float((2 * math.pi ** (k / 2)) / math.gamma(k / 2) * r ** (k - 1))


def sphere_volume(k: int, r: float = 1.0) -> float:
    """Volume of a sphere in `k` embedding dimensions and radius `r`."""
    return float((math.pi ** (k / 2)) / math.gamma(k / 2 + 1) * r**k)


class Sphere(CompactManifold):
    """Class representing a sphere manifold.

    Attributes
    ----------
    dim
        Surface fimension of the sphere.
    r
        Radius of the sphere, by default 1.0.
    """

    def __init__(self, dim: int, r: float = 1.0) -> None:
        super().__init__(dim)
        self.r = r

    @property
    def params(self) -> dict[str, float]:
        """Return the parameters of the sphere."""
        return {"r": self.r}

    @property
    def radius(self) -> float:
        """Return the radius of the sphere, alias for :attr:`r`."""
        return self.r

    @property
    def surface_area(self) -> float:
        return sphere_surface_area(self.embedding_dim, self.r)

    @property
    def volume(self) -> float:
        return sphere_volume(self.embedding_dim, self.r)

    @property
    def circumference(self) -> float:
        return 2 * math.pi * self.r

    @property
    def max_distance(self) -> float:
        """Maximum distance between two points on the sphere."""
        return self.circumference / 2

    def _pdist(self, X: np.ndarray) -> np.ndarray:
        """Compute pairwise distances between points on the sphere.

        Examples
        --------
        >>> sphere = Sphere(2, 3)
        >>> X = sphere.sample_points(5, random_state=42)
        >>> d = sphere.pdist(X, full=False)
        >>> all(d >= 0) and all(d <= sphere.max_distance)
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
        >>> Y = sphere.sample_points(5, random_state=43)
        >>> d = sphere.cdist(X, Y)
        >>> d.shape
        (5, 5)
        >>> bool(np.all(d >= 0) and np.all(d <= sphere.max_distance))
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

    @classmethod
    def from_surface_area(cls, dim: int, area: float) -> Self:
        """Create a sphere with `k`-dimensional surface with given `area`.

        Parameters
        ----------
        dim
            Dimension of the sphere's surface.
        surface_area
            Surface area of the sphere.

        Examples
        --------
        >>> sphere = Sphere.from_surface_area(2, 20)
        >>> sphere.dim
        2
        >>> sphere.surface_area
        20.0
        """
        k = dim + 1  # embedding dimension
        r = float(area * math.gamma(k / 2) / (2 * math.pi ** (k / 2))) ** (1 / (k - 1))
        return cls(dim, r)
