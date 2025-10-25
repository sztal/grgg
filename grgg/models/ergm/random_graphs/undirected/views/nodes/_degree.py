from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from grgg._typing import Integer, Reals
from grgg.statistics import Degree


class RandomGraphDegreeStatistic(Degree):
    def _homogeneous_m1(self, **kwargs: Any) -> Reals:  # noqa
        """Expected degree for homogeneous undirected random graph models.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from grgg import RandomGraph, RandomGenerator
        >>> rng = RandomGenerator(42)
        >>> model = RandomGraph(100, mu=-2)
        >>> kbar = model.nodes.degree()
        >>> kbar.shape
        ()
        >>> kbar.item()
        11.801089
        >>> K = jnp.array(
        ...     [model.sample(rng=rng).A.sum(axis=1).mean() for _ in range(20)]
        ... )
        >>> jnp.isclose(K.mean(), kbar, rtol=1e-1).item()
        True
        >>> K = model.nodes[...].degree()
        >>> K.shape
        (100,)
        >>> jnp.allclose(K, kbar).item()
        True
        """
        return self.model.pairs.probs() * (self.model.n_nodes - 1)

    def _heterogeneous_m1_exact(self, **kwargs: Any) -> Reals:  # noqa
        """Expected degree for heterogeneous undirected random graph models.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from grgg import RandomGraph, RandomGenerator
        >>> rng = RandomGenerator(42)
        >>> mu = rng.normal(100)
        >>> model = RandomGraph(mu.size, mu=mu)
        >>> D = model.nodes.degree()
        >>> D.shape
        (100,)
        >>> K = jnp.column_stack(
        ...     [model.sample(rng=rng).A.sum(axis=1) for _ in range(20)
        ... ]).mean(axis=1)
        >>> jnp.allclose(D, K, rtol=1e-1).item()
        True
        >>> vids = jnp.array([0, 11, 27, 89])
        >>> D = model.nodes[vids].degree()
        >>> D.shape
        (4,)
        >>> jnp.allclose(D, K[vids], rtol=1e-1).item()
        True
        """
        indices = self.nodes.coords[0]

        @eqx.filter_checkpoint
        @eqx.filter_jit
        def node_degree(i: Integer) -> jnp.ndarray:
            return self.model.pairs[i].probs().sum(-1)

        return jax.lax.map(node_degree, indices, batch_size=self.batch_size)
