import math

import numpy as np
from geomstats.geometry.hypersphere import Hypersphere
from scipy.spatial.distance import cdist, pdist, squareform


class Sphere(Hypersphere):
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

    def surface_area(self, radius: float = 1.0) -> float:
        """Compute the surface area of the sphere.

        Parameters
        ----------
        radius
            Radius of the sphere, by default 1.0.
        """
        k = self.embedding_space.dim
        return float((2 * math.pi ** (k / 2)) / math.gamma(k / 2) * radius ** (k - 1))

    def volume(self, radius: float = 1.0) -> float:
        """Compute the volume of the sphere.

        Parameters
        ----------
        radius
            Radius of the sphere, by default 1.0.
        """
        k = self.embedding_space.dim
        return float((math.pi ** (k / 2)) / math.gamma(k / 2 + 1) * radius**k)

    def radius(self, area: float) -> float:
        """Compute the radius of the sphere given its surface area.

        Parameters
        ----------
        area
            Surface area of the sphere.
        """
        k = self.embedding_space.dim
        return float(
            (area * math.gamma(k / 2) / (2 * math.pi ** (k / 2))) ** (1 / (k - 1))
        )
