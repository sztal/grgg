from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp

from grgg._typing import Integer, Real, Reals
from grgg.models.ergm.abc import AbstractErgmFunctions
from grgg.utils.compute import fori

if TYPE_CHECKING:
    from .model import AbstractRandomGraph
    from .views import AbstractRandomGraphNodeView

__all__ = ("AbstractRandomGraphFunctions",)


class AbstractRandomGraphFunctions(AbstractErgmFunctions):
    """Abstract base class for random graph model functions.

    Examples
    --------
    Check analytically free energy computations for Erdős-Rényi model.
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from grgg import RandomGraph, RandomGenerator
    >>> rng = RandomGenerator(17)
    >>> n = 100
    >>> model = RandomGraph(n, mu=-2.0)
    >>> theta = model.mu.theta
    >>> jnp.isclose(model.edge_density(), jax.nn.sigmoid(-theta)).item()
    True
    >>> jnp.isclose(model.pairs.probs(), model.edge_density()).item()
    True

    Model free energy.
    >>> F1 = n*(n-1)/2 * jax.nn.log_sigmoid(theta)
    >>> F2 = model.free_energy()
    >>> jnp.isclose(F1, F2).item()
    True

    Check consistency between model likelihood computations
    using the generic ERGM formula and edge-factorized formula.
    >>> p = model.pairs.probs()
    >>> S = model.sample(rng=rng)
    >>> A = S.A.toarray()[jnp.tril_indices(n, k=-1)]
    >>> ll1 = jnp.where(A == 1, jnp.log(p), jnp.log1p(-p)).sum()
    >>> F = model.free_energy()
    >>> theta = -model.mu.data
    >>> H = theta * A.sum()
    >>> ll2 = -H + F
    >>> jnp.isclose(ll1, ll2).item()
    True

    Now the same, but using model method for Hamiltonian.
    >>> H = model.hamiltonian(S.A)
    >>> ll3 = -H + F
    >>> jnp.isclose(ll1, ll3).item()
    True

    Check edge free energy computations:
    >>> model = RandomGraph(n, mu=rng.normal(n) - 2.5)
    >>> couplings = model.pairs.couplings()
    >>> efe0 = jax.nn.log_sigmoid(couplings)
    >>> efe1 = model.pairs.free_energy()
    >>> jax.numpy.allclose(efe0, efe1).item()
    True

    Check gradient computations of free energy:
    >>> def fe(model): return model.free_energy()
    >>> def fe_naive(model): return model.pairs.free_energy().sum() / 2
    >>> grad = jax.grad(fe)(model)
    >>> grad_naive = jax.grad(fe_naive)(model)
    >>> jnp.allclose(grad.mu.data, grad_naive.mu.data).item()
    True

    Check theoretical consistency of free energy gradients
    in undirected soft configuration model.
    >>> jnp.allclose(grad.mu.theta, model.nodes.degree(), rtol=1e-3).item()
    True
    """

    @classmethod
    @eqx.filter_jit
    def couplings(
        cls, params: "AbstractRandomGraph.Parameters", *args: Any, **kwargs: Any
    ) -> Reals:
        """Compute the couplings of the model."""
        raise NotImplementedError

    @singledispatchmethod
    @classmethod
    def probs(
        cls,
        params: "AbstractRandomGraph.Parameters",
        *args: Any,
        **kwargs: Any,
    ) -> Reals:
        """Compute edge probabilities."""
        return cls._probs_params(params, *args, **kwargs)

    @probs.register
    @classmethod
    def _(cls, couplings: jnp.ndarray, *args: Any, **kwargs: Any) -> Reals:
        """Compute edge probabilities from couplings."""
        return cls._probs_couplings(couplings, *args, **kwargs)

    @singledispatchmethod
    @classmethod
    def edge_free_energy(
        cls, params: "AbstractRandomGraph.Parameters", *args: Any, **kwargs: Any
    ) -> Reals:
        """Compute the edge free energy."""
        return cls._edge_free_energy_params(params, *args, **kwargs)

    @edge_free_energy.register
    @classmethod
    def _(cls, couplings: jnp.ndarray) -> Reals:
        """Compute edge free energy from couplings."""
        return cls._edge_free_energy_couplings(couplings)

    @classmethod
    @eqx.filter_jit
    def node_free_energy(
        cls, nodes: "AbstractRandomGraphNodeView", *args: Any, **kwargs: Any
    ) -> Reals:
        """Compute the free energy contributions from nodes."""
        if nodes.model.is_heterogeneous:
            return cls._node_free_energy_heterogeneous(nodes, *args, **kwargs)
        return cls._node_free_energy_homogeneous(nodes, *args, **kwargs)

    @classmethod
    @eqx.filter_jit
    def free_energy(
        cls, model: "AbstractRandomGraph", *args: Any, **kwargs: Any
    ) -> Real:
        """Compute the free energy of the model."""
        if model.is_heterogeneous:
            return cls._free_energy_heterogeneous(model, *args, **kwargs)
        return cls._free_energy_homogeneous(model, *args, **kwargs)

    # Internals ----------------------------------------------------------------------

    @classmethod
    @eqx.filter_jit
    def _probs_couplings(cls, couplings: Reals, *, log: bool = False) -> Reals:
        """Compute edge probabilities from couplings."""
        if log:
            return jax.nn.log_sigmoid(-couplings)
        return jax.nn.sigmoid(-couplings)

    @classmethod
    @eqx.filter_jit
    def _probs_params(
        cls,
        params: "AbstractRandomGraph.Parameters",
        *args: Any,
        log: bool = False,
        **kwargs: Any,
    ) -> Reals:
        couplings = cls.couplings(params, *args, **kwargs)
        return cls._probs_couplings(couplings, log=log)

    @classmethod
    @eqx.filter_jit
    def _edge_free_energy_couplings(cls, couplings: Reals) -> Reals:
        """Compute the edge free energy from couplings."""
        return jax.nn.log_sigmoid(couplings)

    @classmethod
    @eqx.filter_jit
    def _edge_free_energy_params(
        cls, params: "AbstractRandomGraph.Parameters.Data", *args: Any, **kwargs: Any
    ) -> Reals:
        """Compute the edge free energy from model parameters."""
        couplings = cls.couplings(params, *args, **kwargs)
        return cls._edge_free_energy_couplings(couplings)

    @classmethod
    @eqx.filter_jit
    def _free_energy_homogeneous(
        cls, model: "AbstractRandomGraph", *args: Any, **kwargs: Any
    ) -> Real:
        """Compute the free energy of a homogeneous model."""
        fe = cls._node_free_energy_homogeneous(model.nodes[0], *args, **kwargs)[0]
        return fe * model.n_nodes

    @classmethod
    @eqx.filter_jit
    def _free_energy_heterogeneous(
        cls, model: "AbstractRandomGraph", *args: Any, **kwargs: Any
    ) -> Real:
        """Compute the free energy of a heterogeneous model."""
        return _free_energy_heterogeneous(model, *args, **kwargs)

    @classmethod
    @eqx.filter_jit
    def F_i(
        cls, model: "AbstractRandomGraph", i: Integer, *args: Any, **kwargs: Any
    ) -> Real:
        """Compute the free energy contribution from node i."""
        return model.pairs[i].free_energy(*args, **kwargs).sum()

    @classmethod
    @eqx.filter_jit
    def _node_free_energy_homogeneous(
        cls, nodes: "AbstractRandomGraphNodeView", *args: Any, **kwargs: Any
    ) -> Reals:
        """Compute the free energy contributions from nodes in a homogeneous model."""
        n = nodes.model.n_nodes
        if n < 2:
            fe = nodes.model.pairs[0, 0].free_energy(*args, **kwargs)
        else:
            fe = nodes.model.pairs[1, 0].free_energy(*args, **kwargs)
        return jnp.full((nodes.size,), fe * (n - 1))

    @classmethod
    def _node_free_energy_heterogeneous(
        cls, nodes: "AbstractRandomGraphNodeView", *args: Any, **kwargs: Any
    ) -> Reals:
        """Compute the free energy contribution from node i in a heterogeneous model."""
        return jax.lax.map(
            lambda i: cls.F_i(nodes.model, i, *args, **kwargs), nodes.coords[0]
        )


# Pure function internals for heterogeneous free energy ------------------------------


@eqx.filter_custom_vjp
@eqx.filter_jit
def _free_energy_heterogeneous(
    model: "AbstractRandomGraph", *args: Any, **kwargs: Any
) -> Real:
    """Compute the free energy of a heterogeneous model."""

    @fori(0, model.n_nodes, init=0.0)
    def fe(i: Integer, carry: Real) -> Real:
        return carry + model.functions.F_i(model, i, *args, **kwargs)

    return fe


@_free_energy_heterogeneous.def_fwd
@eqx.filter_jit
def _free_energy_heterogeneous_fwd(
    _, model: "AbstractRandomGraph", *args: Any, **kwargs: Any
) -> tuple[Real, None]:
    """Forward pass for custom VJP of free energy."""
    return _free_energy_heterogeneous(model, *args, **kwargs), None


@_free_energy_heterogeneous.def_bwd
@eqx.filter_jit
def _free_energy_heterogeneous_bwd(
    _,
    g_out: Real,
    __,
    model: "AbstractRandomGraph",
    *args: Any,
    **kwargs: Any,
) -> "AbstractRandomGraph":
    """Backward pass for custom VJP of free energy."""
    # Initialize gradients with zeros matching the model structure
    init_grads = jax.tree_util.tree_map(jnp.zeros_like, model)
    # Pre-compile the gradient function for a single index
    grad_fn = jax.grad(model.functions.F_i, argnums=0)

    @fori(0, model.n_nodes, init=init_grads)
    def gradient(i: Integer, carry: "AbstractRandomGraph") -> "AbstractRandomGraph":
        # Compute gradient for the i-th pair/node
        g_i = grad_fn(model, i, *args, **kwargs)
        # Accumulate gradients
        return jax.tree_util.tree_map(jnp.add, carry, g_i)

    # Apply the chain rule:
    # multiply accumulated grads by the output gradient (g_out)
    return jax.tree_util.tree_map(lambda g: g * g_out, gradient)
