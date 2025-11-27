from typing import Any

import equinox as eqx
import jax.numpy as jnp

from grgg.manifolds import Manifold

__all__ = ("AbstractGeometricGraph",)


class AbstractGeometricGraph:
    """Abstract base class for geometric graphs."""

    manifold: eqx.AbstractVar[Manifold]

    def sample_points(self, *args: Any, **kwargs: Any) -> jnp.ndarray:
        """Sample points from the manifold."""
        return self.nodes.sample_points(*args, **kwargs)

    def sample_pmatrix(self, *args: Any, **kwargs: Any) -> jnp.ndarray:
        """Sample matrix of edge probabilities from the model."""
        return self.nodes.sample_pmatrix(*args, **kwargs)
