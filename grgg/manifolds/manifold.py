from abc import ABC, abstractmethod
from typing import Any, Self

import numpy as np
from scipy.spatial.distance import squareform

from grgg.utils import random_generator


class Manifold(ABC):
    """Abstract base class for manifolds.

    Attributes
    ----------
    dim
        Dimension of the manifold.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def __repr__(self) -> str:
        cn = self.__class__.__name__
        params = [
            f"{k}={v:f.3}" if isinstance(v, float) else f"{k}={v}"
            for k, v in self.params.items()
        ]
        params = [str(self.dim), *params]
        return f"{cn}({",".join(params)})"

    def __copy__(self) -> Self:
        return self.__class__(self.dim, **self.params)

    @property
    @abstractmethod
    def params(self) -> dict[str, float]:
        return {}

    @property
    def embedding_dim(self) -> int:
        return self.dim + 1

    @property
    @abstractmethod
    def surface_area(self) -> float:
        pass

    @property
    @abstractmethod
    def volume(self) -> float:
        pass

    def pdist(self, X: np.ndarray, *, full: bool = False) -> np.ndarray:
        """Compute pairwise distances between points on the manifold."""
        dist = self._pdist(X)
        if full:
            dist = squareform(dist)
        return dist

    @abstractmethod
    def _pdist(self, X: np.ndarray) -> np.ndarray:
        """Internal method to compute pairwise distances."""

    def cdist(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute pairwise distances between two sets of points on the manifold."""
        return self._cdist(X, Y)

    @abstractmethod
    def _cdist(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Internal method to compute pairwise distances between two sets of points."""

    def sample_points(
        self, n: int, *, random_state: np.random.Generator | int | None = None
    ) -> np.ndarray:
        """Sample points uniformly from the manifold.

        Parameters
        ----------
        n
            Number of points to sample.
        random_state
            Random state for reproducibility, can be an integer seed
            or a numpy random generator.
        """
        if n <= 0:
            errmsg = "'n' must be positive"
            raise ValueError(errmsg)
        rng = random_generator(random_state)
        return self._sample_points(n, rng)

    @abstractmethod
    def _sample_points(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Internal method to sample points uniformly from the manifold."""

    def copy(self) -> Self:
        """Create a copy of the manifold."""
        return self.__copy__()

    @classmethod
    @abstractmethod
    def from_surface_area(
        cls, dim: int, area: float, *args: Any, **kwargs: Any
    ) -> Self:
        """Create a manifold instance from its surface area."""


class CompactManifold(Manifold):
    """Abstract base class for compact manifolds.

    A compact manifold is a manifold that is both closed and bounded.
    This class inherits from `Manifold` and can be used to define
    specific compact manifolds.
    """

    @property
    @abstractmethod
    def max_distance(self) -> float:
        """Maximum distance between any two points on the manifold."""
