from abc import abstractmethod
from collections.abc import Callable
from typing import Self

import equinox as eqx
import jax.numpy as jnp

from grgg._typing import Matrix, Scalar, Vector
from grgg.abc import AbstractModule
from grgg.random import RandomGenerator
from grgg.utils import pairwise

from .functions import (
    ManifoldCosineLawFunction,
    ManifoldDistanceDensityFunction,
    ManifoldMetricFunction,
    ManifoldVolumeFunction,
)

DistanceFunctionT = Callable[[Matrix, Matrix | None, ...], Vector | Matrix]

__all__ = ("Manifold", "CompactManifold")


class ManifoldComputeNamespace(AbstractModule):
    """Namespace for manifold computations.

    Attributes
    ----------
    volume
        Manifold volume function.
    metric
        Manifold metric function.
    distance_density
        Manifold distance density function.
    """

    volume: ManifoldVolumeFunction
    metric: ManifoldMetricFunction
    distance_density: ManifoldDistanceDensityFunction
    cosine_law: ManifoldCosineLawFunction

    def equals(self, other: object) -> bool:
        return (
            super().equals(other)
            and self.volume.equals(other.volume)
            and self.metric.equals(other.metric)
            and self.distance_density.equals(other.distance_density)
            and self.cosine_law.equals(other.cosine_law)
        )

    @classmethod
    def from_manifold(cls, manifold: "Manifold") -> Self:
        """Create a compute namespace for the specified manifold."""
        volume = ManifoldVolumeFunction.from_manifold(manifold)
        metric = ManifoldMetricFunction.from_manifold(manifold)
        distance_density = ManifoldDistanceDensityFunction.from_manifold(manifold)
        cosine_law = ManifoldCosineLawFunction.from_manifold(manifold)
        return cls(
            volume=volume,
            metric=metric,
            distance_density=distance_density,
            cosine_law=cosine_law,
        )


class Manifold(AbstractModule):
    """Abstract base class for manifolds.

    Attributes
    ----------
    dim
        Dimension of the manifold.
    """

    dim: int = eqx.field(static=True, converter=int)
    compute: ManifoldComputeNamespace = eqx.field(repr=False)
    _distances: DistanceFunctionT = eqx.field(static=True, repr=False, init=False)

    def __init__(
        self, dim: int, *, compute: ManifoldComputeNamespace | None = None
    ) -> None:
        self.dim = int(dim)
        if compute is None:
            compute = ManifoldComputeNamespace.from_manifold(self)
        self.compute = compute
        self._distances = pairwise(self.metric)

    def __check_init__(self) -> None:
        if self.dim < 0:
            errmsg = "'dim' must be a non-negative integer"
            raise ValueError(errmsg)

    def __repr__(self) -> str:
        params = self._repr_params()
        if params:
            params = ", " + params
        return f"{self.__class__.__name__}({self.dim}{params})"

    @abstractmethod
    def _repr_params(self) -> str:
        """Parameters to include in the `__repr__` string."""

    @property
    def embedding_dim(self) -> int:
        return self.dim + 1

    @property
    @abstractmethod
    def volume(self) -> float:
        """Volume of the manfiold surface."""

    @property
    @abstractmethod
    def linear_size(self) -> float:
        """Linear size of the manifold."""

    @abstractmethod
    def with_volume(cls, volume: float) -> Self:
        """Create a manifold instance with the specified volume."""

    @abstractmethod
    def equals(self, other: object) -> bool:
        return (
            super().equals(other)
            and self.dim == other.dim
            and self.compute.equals(other.compute)
        )

    def metric(self, x: Vector, y: Vector) -> Scalar:
        """Geodesic distance between two points on the manifold."""
        return self.compute.metric(x, y, self.dim, self.linear_size)

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

    def sample_points(
        self, n: int, *, rng: RandomGenerator | int | None = None
    ) -> Matrix:
        """Sample points uniformly from the manifold.

        Parameters
        ----------
        n
            Number of points to sample.
        rng
            Random generator or an object interpretable as a seed.

        Returns
        -------
        Matrix
            Sampled points on the manifold, shape (n, dim+1).
        """
        if n <= 0:
            errmsg = "'n' must be positive"
            raise ValueError(errmsg)
        rng = RandomGenerator.from_seed(rng)
        return self._sample_points(n, rng)

    @abstractmethod
    def _sample_points(self, n: int, rng: RandomGenerator) -> Matrix:
        """Implementation of point sampling."""

    def distance_density(self, g: float) -> float:
        """Probability density of the distance between two random points.

        Parameters
        ----------
        g
            Geodesic distance between two points.
        """
        return self.compute.distance_density(g, self.dim, self.linear_size)

    def cosine_law(
        self, theta: jnp.ndarray, g_ij: jnp.ndarray, g_ik: jnp.ndarray
    ) -> jnp.ndarray:
        """Cosine law to compute the third side of a triangle on the manifold.

        Parameters
        ----------
        theta
            Angle between sides `(i, j)` and `(i, k)` geodesics.
        g_ij
            Length of `(i, j)` geodesic.
        g_ik
            Length of `(i, k)` geodesic.

        Returns
        -------
        g_jk
            Length of `(j, k)` geodesic.
        """
        return self.compute.cosine_law(theta, g_ij, g_ik, self.linear_size)


class CompactManifold(Manifold):
    """Abstract base class for compact manifolds."""

    @property
    @abstractmethod
    def diameter(self) -> float:
        """Diameter of the manifold."""
