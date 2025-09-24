from abc import abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Self

import jax.numpy as jnp

from grgg._typing import Matrix, Vector
from grgg.functions import AbstractFunction

if TYPE_CHECKING:
    from .manifold import Manifold

DistanceFunctionT = Callable[[Matrix, Matrix | None, ...], Vector | Matrix]


class ManifoldFunction(AbstractFunction):
    """Base class for manifold functions."""

    def __init_subclass__(cls) -> None:
        if not hasattr(cls, "registry"):
            cls.registry = {}
        if (manifold_type := getattr(cls, "manifold_type", None)) is not None:
            cls.registry[manifold_type.__qualname__] = cls

    @classmethod
    def from_manifold(cls, manifold: "Manifold", *args: Any, **kwargs: Any) -> Self:
        """Get the manifold function for a specific manifold instance."""
        manifold_type = type(manifold).__qualname__
        if manifold_type not in cls.registry:
            errmsg = (
                f"No '{cls.__name__}' registered for manifold type '{manifold_type}'"
            )
            raise TypeError(errmsg)
        return cls.registry[manifold_type](*args, **kwargs)


class ManifoldVolumeFunction(ManifoldFunction):
    """Manifold volume function."""

    def __call__(self, dim: int, linear_size: float) -> jnp.ndarray:
        """Compute the volume."""
        return self.unit(dim) * self.kernel(dim, linear_size)

    @abstractmethod
    def kernel(self, dim: int, linear_size: float) -> jnp.ndarray:
        """Kernel factor in the volume expression."""

    @abstractmethod
    def unit(self, dim: int) -> float:
        """Compute volume of the manifold with unit linear size."""


class ManifoldMetricFunction(ManifoldFunction):
    """Manifold metric function."""

    def __call__(
        self, x: Vector, y: Vector, dim: int, linear_size: float = 1.0
    ) -> jnp.ndarray:
        """Compute the metric (geodesic distance)."""
        x = x / linear_size
        y = y / linear_size
        return self.unit(x, y, dim) * linear_size

    @abstractmethod
    def unit(self, x: Vector, y: Vector, dim: int) -> jnp.ndarray:
        """Metric on the manifold with unit linear size."""


class ManifoldDistanceDensityFunction(ManifoldFunction):
    """Manifold distance density function."""

    def __call__(self, g: float, dim: int, linear_size: float) -> jnp.ndarray:
        """Compute the distance density."""
        return jnp.maximum(
            0.0, self.constant(dim, linear_size) * self.kernel(g, dim, linear_size)
        )

    @abstractmethod
    def kernel(self, g: jnp.ndarray, dim: int, linear_size: float) -> jnp.ndarray:
        """Kernel factor in the distance density expression."""

    @abstractmethod
    def constant(self, dim: int, linear_size: float) -> float:
        """Constant factor in the distance density expression."""


class ManifoldCosineLawFunction(ManifoldFunction):
    """Manifold cosine law function."""

    def __call__(
        self,
        theta: jnp.ndarray,
        g_ij: jnp.ndarray,
        g_ik: jnp.ndarray,
        linear_size: float,
    ) -> jnp.ndarray:
        """Compute the angle geodesic distance between points j and k given the
        distances to point i and the angle between the geodesics at i.
        """
        angle = self.unit(theta, g_ij / linear_size, g_ik / linear_size)
        return angle * linear_size

    @abstractmethod
    def unit(
        self, theta: jnp.ndarray, theta_ij: jnp.ndarray, theta_ik: jnp.ndarray
    ) -> float:
        """Cosine law on the manifold with unit linear size."""
