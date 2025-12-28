from typing import TYPE_CHECKING, Any

import jax

from grgg._typing import Integer, Real, Reals
from grgg.models.base.ergm import AbstractErgmFunctions
from grgg.models.base.model import AbstractParameters

if TYPE_CHECKING:
    from .model import AbstractRandomGraph

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
    >>> H = model.mu.theta * A.sum()
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
    def couplings(cls, params: AbstractParameters, *args: Any, **kwargs: Any) -> Reals:
        """Compute the couplings of the model."""
        raise NotImplementedError

    @classmethod
    def probs(cls, couplings: Reals, log: bool = False) -> Reals:
        """Compute edge probabilities from couplings."""
        if log:
            return jax.nn.log_sigmoid(-couplings)
        return jax.nn.sigmoid(-couplings)

    @classmethod
    def edge_free_energy(cls, couplings: Reals) -> Reals:
        """Compute edge free energy from coupling(s)."""
        return jax.nn.log_sigmoid(couplings)

    @classmethod
    def node_free_energy(
        cls,
        model: "AbstractRandomGraph",
        i: int | Integer,
        *args: Any,
        normalize: bool = False,
        **kwargs: Any,
    ) -> Real:
        """Compute the free energy contribution from node i.

        Parameters
        ----------
        model
            The model to compute the node free energy for.
        i
            The node index.
        normalize
            Whether to normalize the free energy by the number of nodes.
        *args, **kwargs
            Additional arguments.
        """
        if model.is_homogeneous:
            ij = (0, 0) if (n := model.n_nodes) < 2 else (1, 0)
            fe = model.pairs[ij].free_energy(*args, **kwargs) * (n - 1)
        else:
            fe = model.pairs[i].free_energy(*args, **kwargs).sum(axis=0)
        fe = fe / model.n_nodes if normalize else fe
        return fe if model.is_directed else fe / 2
