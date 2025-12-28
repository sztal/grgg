from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

from grgg._typing import Integer, Integers, Real, Reals, RealVector
from grgg.statistics.motifs import QHeadMotif

if TYPE_CHECKING:
    from grgg.models.random_graph import RandomGraph


class RandomGraphQHeadMotif(QHeadMotif):
    """Q-head motif statistic for undirected random graphs.

    Examples
    --------
    Homogeneous case.
    >>> import jax.numpy as jnp
    >>> from grgg import RandomGraph, RandomGenerator
    >>> rng = RandomGenerator(17)
    >>> model = RandomGraph(100, mu=-2)
    >>> qhead = model.nodes.motifs.qhead()
    >>> QH = jnp.array(
    ...     [model.sample(rng=rng).struct.census().qh.mean() for _ in range(20)]
    ... )
    >>> jnp.isclose(qhead, QH.mean(), rtol=5e-2).item()
    True

    Heterogeneous case (exact).
    >>> import jax.numpy as jnp
    >>> from grgg import RandomGraph, RandomGenerator
    >>> rng = RandomGenerator(42)
    >>> model = RandomGraph(100, mu=rng.normal(100) - 1.2)
    >>> qhead = model.nodes.motifs.qhead()
    >>> QH = jnp.column_stack(
    ...     [model.sample(rng=rng).struct.census().qh.to_numpy() for _ in range(20)]
    ... ).mean(axis=-1)
    >>> jnp.isclose(qhead.mean(), QH.mean(), rtol=1e-1).item()
    True
    >>> (jnp.corrcoef(qhead, QH)[0, 1] > 0.99).item()
    True

    Heterogeneous case (Monte Carlo).
    >>> import jax.numpy as jnp
    >>> from grgg import RandomGraph, RandomGenerator
    >>> rng = RandomGenerator(42)
    >>> n = 500
    >>> model = RandomGraph(n, mu=rng.normal(n) - 2.5)
    >>> qh0 = jnp.log(model.nodes.motifs.qhead())
    >>> qh1 = jnp.log(model.nodes.motifs.qhead(mc=50, rng=rng))
    >>> err = jnp.linalg.norm(qh0 - qh1) / jnp.linalg.norm(qh0)
    >>> (err < 0.02).item()
    True
    >>> cor = jnp.corrcoef(qh0, qh1)[0, 1]
    >>> (cor > 0.99).item()
    True
    """

    @staticmethod
    def _kernel_m1_exact(
        model: "RandomGraph", i: Integer, j: Integer, degree: RealVector
    ) -> Real:
        p_ij = model.pairs[i, j].probs()
        p_ik = model.pairs[i].probs().at[j].set(0.0)
        p_jk = model.pairs[j].probs().at[i].set(0.0)
        d_k = degree.at[jnp.array([i, j])].set(0.0)
        return p_ij * jnp.sum(p_jk * (d_k - p_ik - p_jk))

    @staticmethod
    def _kernel_m1_mc(
        model: "RandomGraph", i: Integer, j: Integer, degree: RealVector
    ) -> Real:
        p_ik = model.pairs[i].probs().at[j].set(0.0)
        p_jk = model.pairs[j].probs().at[i].set(0.0)
        d_k = degree.at[jnp.array([i, j])].set(0.0)
        return jnp.sum(p_jk * (d_k - p_ik - p_jk))

    @staticmethod
    def _mc_weights(model: "RandomGraph", depth: int, vids: Integers) -> Reals:
        if depth == 0:
            return model.pairs[vids[depth]].probs()
        raise NotImplementedError

    def _homogeneous_m1(self, **kwargs: Any) -> Reals:  # noqa
        n = self.model.n_nodes
        p = self.model.pairs.probs()
        return (n - 1) * (n - 2) * (n - 3) * p**3

    def _heterogeneous_m1_exact(self) -> Reals:
        degree = self.model.nodes.degree()
        return self.iteration(
            order=1,
            kernel=lambda m, i, j: self._kernel_m1_exact(m, i, j, degree),
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
