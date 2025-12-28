from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from grgg._typing import Integer, Integers, Real, Reals
from grgg.statistics.motifs import TWedgeMotif
from grgg.utils.random import RandomGenerator

if TYPE_CHECKING:
    from grgg.models.random_graph import RandomGraph

__all__ = ("RandomGraphTWedgeMotif",)


class RandomGraphTWedgeMotif(TWedgeMotif):
    """T-wedge motif statistic for undirected random graphs.

    Examples
    --------
    Homogeneous case.
    >>> import jax.numpy as jnp
    >>> from grgg import RandomGraph, RandomGenerator
    >>> rng = RandomGenerator(17)
    >>> model = RandomGraph(100, mu=-2)
    >>> twedge = model.nodes.motifs.twedge()
    >>> tw = jnp.array(
    ...     [model.sample(rng=rng).struct.census().tw.mean() for _ in range(20)]
    ... )
    >>> jnp.isclose(twedge, tw.mean(), rtol=5e-2).item()
    True

    Heterogeneous case (exact).
    >>> import jax.numpy as jnp
    >>> from grgg import RandomGraph, RandomGenerator
    >>> rng = RandomGenerator(42)
    >>> model = RandomGraph(100, mu=rng.normal(100) - 1.2)
    >>> twedge = model.nodes.motifs.twedge()
    >>> tw = jnp.column_stack(
    ...     [model.sample(rng=rng).struct.census().tw.to_numpy() for _ in range(20)]
    ... ).mean(axis=-1)
    >>> jnp.isclose(twedge.mean(), tw.mean(), rtol=1e-1).item()
    True
    >>> (jnp.corrcoef(twedge, tw)[0, 1] > 0.99).item()
    True

    Heterogeneous case (Monte Carlo).
    >>> import jax.numpy as jnp
    >>> from grgg import RandomGraph, RandomGenerator
    >>> rng = RandomGenerator(42)
    >>> n = 500
    >>> model = RandomGraph(n, mu=rng.normal(n) - 2.5)
    >>> tw0 = jnp.log(model.nodes.motifs.twedge())
    >>> tw1 = jnp.log(model.nodes.motifs.twedge(mc=50, rng=rng))
    >>> err = jnp.linalg.norm(tw0 - tw1) / jnp.linalg.norm(tw0)
    >>> (err < 0.01).item()
    True
    >>> cor = jnp.corrcoef(tw0, tw1)[0, 1]
    >>> (cor > 0.99).item()
    True
    """

    @staticmethod
    def _kernel_m1_exact(model: "RandomGraph", i: Integer) -> Real:
        p_ij = model.pairs[i].probs()
        d_i = p_ij.sum()
        return jnp.sum(p_ij * (d_i - p_ij))

    @staticmethod
    def _kernel_m1_mc(model: "RandomGraph", i: Integer, key: Integers, mc: int) -> Real:
        p_ij = model.pairs[i].probs()
        d_i = p_ij.sum()
        key = jax.random.fold_in(key, i)
        p_ij = jax.random.choice(key, p_ij, p=p_ij, shape=(mc,), replace=True)
        return d_i * jnp.mean(d_i - p_ij)

    def _homogeneous_m1(self) -> Reals:
        n = self.model.n_nodes
        p = self.model.pairs.probs()
        return (n - 1) * (n - 2) * p**2

    def _heterogeneous_m1_exact(self) -> Reals:
        return self.iteration(
            order=0,
            kernel=self._kernel_m1_exact,
            unique=False,
        ).map(self.model, self.nodes.coords[0])

    def _heterogeneous_m1_monte_carlo(self) -> Reals:
        key = RandomGenerator.make_key(self.key)
        return self.iteration(
            order=0,
            kernel=lambda m, i: self._kernel_m1_mc(m, i, key, self.mc),
            unique=False,
        ).map(self.model, self.nodes.coords[0])
