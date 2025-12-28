from typing import TYPE_CHECKING

import jax.numpy as jnp

from grgg._typing import Integer, Integers, Real, Reals, RealVector
from grgg.statistics.motifs import THeadMotif

if TYPE_CHECKING:
    from grgg.models.random_graph import RandomGraph

__all__ = ("RandomGraphTHeadMotif",)


class RandomGraphTHeadMotif(THeadMotif):
    """T-head motif statistic for undirected random graphs.

    Examples
    --------
    Homogeneous case.
    >>> import jax.numpy as jnp
    >>> from grgg import RandomGraph, RandomGenerator
    >>> rng = RandomGenerator(17)
    >>> model = RandomGraph(100, mu=-2)
    >>> thead = model.nodes.motifs.thead()
    >>> TH = jnp.array(
    ...     [model.sample(rng=rng).struct.census().th.mean() for _ in range(20)]
    ... )
    >>> jnp.isclose(thead, TH.mean(), rtol=5e-2).item()
    True

    Heterogeneous case (exact).
    >>> import jax.numpy as jnp
    >>> from grgg import RandomGraph, RandomGenerator
    >>> rng = RandomGenerator(42)
    >>> model = RandomGraph(100, mu=rng.normal(100) - 1.2)
    >>> thead = model.nodes.motifs.thead()
    >>> TH = jnp.column_stack(
    ...     [model.sample(rng=rng).struct.census().th.to_numpy() for _ in range(20)]
    ... ).mean(axis=-1)
    >>> jnp.isclose(thead.mean(), TH.mean(), rtol=1e-1).item()
    True
    >>> (jnp.corrcoef(thead, TH)[0, 1] > 0.99).item()
    True

    Heterogeneous case (Monte Carlo).
    >>> import jax.numpy as jnp
    >>> from grgg import RandomGraph, RandomGenerator
    >>> rng = RandomGenerator(42)
    >>> n = 500
    >>> model = RandomGraph(n, mu=rng.normal(n) - 2.5)
    >>> th0 = jnp.log(model.nodes.motifs.thead())
    >>> th1 = jnp.log(model.nodes.motifs.thead(mc=100, rng=rng))
    >>> err = jnp.linalg.norm(th1 - th0) / jnp.linalg.norm(th0)
    >>> (err < 0.02).item()
    True
    >>> cor = jnp.corrcoef(th0, th1)[0, 1]
    >>> (cor > 0.99).item()
    True
    """

    @staticmethod
    def _kernel_m1_exact(model: "RandomGraph", i: Integer, degree: RealVector) -> Real:
        j = jnp.delete(jnp.arange(model.n_nodes), i, assume_unique_indices=True)
        p_ij = model.pairs[i, j].probs()
        return jnp.sum(p_ij * (degree[j] - p_ij))

    @staticmethod
    def _kernel_m1_mc(
        model: "RandomGraph", i: Integer, j: Integer, degree: RealVector
    ) -> Real:
        p_ij = model.pairs[i, j].probs()
        return degree[j] - p_ij

    @staticmethod
    def _mc_weights(model: "RandomGraph", depth: int, vids: Integers) -> Reals:
        if depth == 0:
            return model.pairs[vids[depth]].probs()
        raise NotImplementedError

    def _homogeneous_m1(self) -> Reals:
        n = self.model.n_nodes
        p = self.model.pairs.probs()
        return (n - 1) * (n - 2) * p**2

    def _heterogeneous_m1_exact(self) -> Reals:
        degree = self.model.nodes.degree()
        return self.iteration(
            order=0,
            kernel=lambda m, i: self._kernel_m1_exact(m, i, degree),
            unique=True,
        ).map(self.model, self.nodes.coords[0])

    def _heterogeneous_m1_monte_carlo(self) -> Reals:
        degree = self.model.nodes.degree()
        return self.iteration(
            order=1,
            kernel=lambda m, i, j: self._kernel_m1_mc(m, i, j, degree),
            unique=True,
            weights=self._mc_weights,
        ).map(self.model, self.nodes.coords[0])
