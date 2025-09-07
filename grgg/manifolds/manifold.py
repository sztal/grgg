from abc import ABC, abstractmethod
from typing import Any, Self

import numpy as np
from scipy.spatial.distance import squareform


class Manifold(ABC):
    """Abstract base class for manifolds.

    Attributes
    ----------
    dim
        Dimension of the manifold.
    """

    def __init__(self, dim: int) -> None:
        if dim < 0:
            errmsg = "'dim' must be a non-negative integer"
            raise ValueError(errmsg)
        self.dim = int(dim)

    def __repr__(self) -> str:
        cn = self.__class__.__name__
        params = self.params
        dim = params.pop("dim")
        params = [
            f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in params.items()
        ]
        params = [str(dim), *params]
        return f"{cn}({", ".join(params)})"

    def __copy__(self) -> Self:
        return self.__class__(**self.params)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Manifold):
            return NotImplemented
        return self.dim == other.dim and self.params == other.params

    def __hash__(self) -> int:
        return hash((self.dim, frozenset(self.params.items())))

    @property
    @abstractmethod
    def params(self) -> dict[str, float]:
        return {"dim": self.dim}

    @property
    def embedding_dim(self) -> int:
        return self.dim + 1

    @property
    @abstractmethod
    def volume(self) -> float:
        """Volume of the manfiold surface."""

    def pdist(self, X: np.ndarray, *, full: bool = False) -> np.ndarray:
        """Compute pairwise geodesic distances between points on the manifold."""
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
        random_state = np.random.default_rng(random_state)
        return self._sample_points(n, random_state)

    @abstractmethod
    def _sample_points(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Internal method to sample points uniformly from the manifold."""

    def copy(self) -> Self:
        """Create a copy of the manifold."""
        return self.__copy__()

    @abstractmethod
    def with_volume(cls, volume: float, *args: Any, **kwargs: Any) -> Self:
        """Create a manifold instance from a target volume."""


class CompactManifold(Manifold):
    """Abstract base class for compact manifolds.

    A compact manifold is a manifold that is both closed and bounded.
    This class inherits from `Manifold` and can be used to define
    specific compact manifolds.
    """

    @property
    @abstractmethod
    def diameter(self) -> float:
        """Maximum geodesic distance between any two points on the manifold."""
