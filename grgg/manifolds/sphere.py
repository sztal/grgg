import math
from typing import Self

import numpy as np
from geomstats.geometry.hypersphere import Hypersphere
from scipy.spatial.distance import cdist, pdist, squareform

__all__ = ("Sphere",)


class Sphere(Hypersphere):
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.dim})"

    def __copy__(self) -> Self:
        return self.__class__(self.dim, intrinsic=self.intrinsic)

    def pdist(self, X: np.ndarray, *, full: bool = True) -> np.ndarray:
        """Compute pairwise distances between points on the sphere."""
        angles = np.arccos(1 - pdist(X, metric="cosine"))
        if full:
            angles = squareform(angles)
        return angles

    def cdist(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute pairwise distances between two sets of points on the sphere."""
        angles = np.arccos(1 - cdist(X, Y, metric="cosine"))
        return angles

    def surface_area(self, radius: float = 1.0, k: int | None = None) -> float:
        """Compute the surface area of the sphere.

        Parameters
        ----------
        radius
            Radius of the sphere, by default 1.0.
        k
            Embedding dimension of the sphere, defaults to `self.embedding_space.dim`.
        """
        k = self.embedding_space.dim if k is None else k
        return float((2 * math.pi ** (k / 2)) / math.gamma(k / 2) * radius ** (k - 1))

    def volume(self, radius: float = 1.0, k: int | None = None) -> float:
        """Compute the volume of the sphere.

        Parameters
        ----------
        radius
            Radius of the sphere, by default 1.0.
        k
            Embedding dimension of the sphere, defaults to `self.embedding_space.dim`.
        """
        k = self.embedding_space.dim if k is None else k
        return float((math.pi ** (k / 2)) / math.gamma(k / 2 + 1) * radius**k)

    def radius(self, area: float | None = None, k: int | None = None) -> float:
        """Compute the radius of the sphere given surface area
        or return the radius of the canonical unit sphere.

        Parameters
        ----------
        area
            Surface area of the sphere.
        k
            Embedding dimension of the sphere, defaults to `self.embedding_space.dim`.
        """
        if area is None:
            return 1.0
        k = self.embedding_space.dim if k is None else k
        return float(
            (area * math.gamma(k / 2) / (2 * math.pi ** (k / 2))) ** (1 / (k - 1))
        )

    def copy(self) -> Self:
        return self.__copy__()
