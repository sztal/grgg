from abc import abstractmethod
from typing import Self

from flax import nnx

from grgg._typing import Matrix, Scalar, Vector
from grgg.abc import AbstractModule
from grgg.utils import pairwise, random_state


class Manifold(AbstractModule):
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
        self._distances = pairwise(self.metric)

    @property
    def embedding_dim(self) -> int:
        return self.dim + 1

    @property
    @abstractmethod
    def volume(self) -> float:
        """Volume of the manfiold surface."""

    @abstractmethod
    def with_volume(cls, volume: float) -> Self:
        """Create a manifold instance with the specified volume."""

    @abstractmethod
    def equals(self, other: object) -> bool:
        return super().equals(other) and self.dim == other.dim

    @abstractmethod
    def metric(self, x: Vector, y: Vector) -> Scalar:
        """Geodesic distance between two points on the manifold."""

    def distances(
        self,
        X: Matrix,
        Y: Matrix | None = None,
        *,
        condensed: bool = True,
    ) -> Vector | Matrix:
        """Compute pairwise distances between points on the manifold.

        Parameters
        ----------
        X
            First set of points, shape (n, dim+1).
        Y
            Second set of points, shape (m, dim+1). If `None`, compute
            pairwise distances within `X`.
        condensed
            If `True` and `Y` is `None`, return a condensed distance matrix,
            shape (n * (n - 1) / 2,). Otherwise, return a full distance matrix,
            shape (n, n) if `Y` is `None`, or (n, m) if `Y` is provided.
        """
        return self._distances(X, Y, condensed=condensed)

    def sample_points(self, n: int, *, rngs: nnx.Rngs | int | None = None) -> Matrix:
        """Sample points uniformly from the manifold.

        Parameters
        ----------
        n
            Number of points to sample.
        rngs
            Random state for reproducibility, can be an integer seed,
            a `nnx.Rngs` object, or `None` for random initialization.

        Returns
        -------
        Matrix
            Sampled points on the manifold, shape (n, dim+1).
        """
        if n <= 0:
            errmsg = "'n' must be positive"
            raise ValueError(errmsg)
        rngs = random_state(rngs)
        return self._sample_points(n, rngs)

    @abstractmethod
    def _sample_points(self, n: int, rngs: nnx.Rngs) -> Matrix:
        """Implementation of point sampling."""


class CompactManifold(Manifold):
    """Abstract base class for compact manifolds."""

    @property
    @abstractmethod
    def diameter(self) -> float:
        """Diameter of the manifold."""
