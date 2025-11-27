from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax

from grgg._typing import Integer, Real, Reals
from grgg.statistics import Degree
from grgg.utils.compute import map

if TYPE_CHECKING:
    from ...model import RandomGraph


class RandomGraphDegree(Degree):
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
        >>> import jax
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

        Check gradients via autodiff.
        >>> def degsum(model): return model.nodes.degree().sum()
        >>> def degsum_naive(model):
        ...     return model.pairs.probs().sum(axis=1).sum()
        >>> grad = jax.grad(degsum)(model)
        >>> grad_naive = jax.grad(degsum_naive)(model)
        >>> jnp.allclose(grad.mu.data, grad_naive.mu.data).item()
        True
        """
        indices = self.nodes.coords[0]

        @map(indices, batch_size=self.batch_size)
        @eqx.filter_jit
        @eqx.filter_checkpoint
        def degree(i: Integer) -> Real:
            return self.f_i(i)

        return degree

    def f_i(self, i: Integer) -> Real:
        """Compute the expected degree of node i."""
        return _node_degree(self.model, i)


@eqx.filter_custom_jvp
@eqx.filter_jit
def _node_degree(model: "RandomGraph", i: Integer) -> Real:
    """Compute the degree of a node in a heterogeneous model."""
    return model.pairs[i].probs().sum()


@_node_degree.def_jvp
@eqx.filter_jit
def _node_degree_jvp(
    primals: tuple["RandomGraph", Integer],
    tangents: tuple["RandomGraph", Any],
) -> tuple[Real, Real]:
    """JVP rule for custom JVP of node degree."""
    model, i = primals
    model_dot, _ = tangents
    primal_out = _node_degree(model, i)
    tangent_out = jax.jvp(lambda m: m.pairs[i].probs().sum(), (model,), (model_dot,))[1]
    return primal_out, tangent_out


# @eqx.filter_custom_vjp
# @eqx.filter_jit
# def _node_degree(model: "RandomGraph", i: Integer) -> Real:
#     """Compute the degree of a node in a heterogeneous model."""
#     return model.pairs[i].probs().sum()


# @_node_degree.def_fwd
# @eqx.filter_jit
# def _node_degree_fwd(_, model: "RandomGraph", i: Integer) -> tuple[Real, None]:
#     """Forward pass for custom VJP of node degree."""
#     return _node_degree(model, i), None


# @_node_degree.def_bwd
# @eqx.filter_jit
# def _node_degree_bwd(
#     _,
#     g_out: Real,
#     __,
#     model: "RandomGraph",
#     i: Integer,
# ) -> "RandomGraph":
#     """Backward pass for custom VJP of node degree."""
#     prob_grad = jax.grad(lambda model, j: model.pairs[i, j].probs().sum(), argnums=0)
#     gradient = prob_grad(model, slice(model.n_nodes))

#     return jax.tree_util.tree_map(lambda g: g_out * g, gradient)
