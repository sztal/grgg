from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import optimistix as optx
from jaxtyping import PyTree

from grgg._typing import Real
from grgg.utils.dispatch import dispatch
from grgg.utils.variables import ArrayBundle

from ..model import AbstractModelFit

if TYPE_CHECKING:
    from .model import AbstractErgm

__all__ = ("LagrangianFit",)


T = TypeVar("T", bound="AbstractErgm")


class LagrangianFit[T, SS](AbstractModelFit[T, SS]):
    """ERGM model fit based on the model Lagrangian.

    Examples
    --------
    >>> import random
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import igraph as ig
    >>> from grgg import RandomGraph
    >>> random.seed(303)
    >>> n = 1000
    >>> G = ig.Graph.Erdos_Renyi(n, p=10/n)

    Homogeneous case.
    >>> solution = RandomGraph(n).fit(G, mu=0).solve()
    >>> model = solution.value
    >>> jnp.isclose(model.edge_density(), G.density()).item()
    True

    Heterogeneous case.
    >>> solution = RandomGraph(n).fit(G, mu="zeros").solve()
    >>> model = solution.value
    >>> expected = model.nodes.degree()
    >>> observed = jnp.array(G.degree())
    >>> jnp.allclose(observed, expected, rtol=1e-3, atol=5e-2).item()
    True
    """

    model: T
    target: ArrayBundle
    method: ClassVar[str] = "lagrangian"
    solver_cls: ClassVar[type[eqx.Module]] = optx.LBFGS

    @dispatch
    def get_tags(self, model: Any) -> frozenset:  # noqa
        """Get solver tags for the given model."""
        return frozenset({lx.negative_semidefinite_tag})

    @property
    def sufficient_statistics(self) -> ArrayBundle:
        """Sufficient statistics used for fitting (alias for `self.target`)."""
        return self.target

    def hamiltonian(
        self,
        model: "AbstractErgm | None" = None,
        *,
        normalize: bool = False,
    ) -> Real:
        """Compute the Hamiltonian of the model.

        Parameters
        ----------
        model
            The model to compute the Hamiltonian for.
            If `None`, uses the model associated with this fit.
        normalize
            Whether to normalize the Hamiltonian by the number of nodes.

        Examples
        --------
        >>> import random
        >>> import igraph as ig
        >>> import jax.numpy as jnp
        >>> from grgg import RandomGenerator, RandomGraph
        >>> random.seed(303)
        >>> rng = RandomGenerator(17)
        >>> n = 100
        >>> G = ig.Graph.Erdos_Renyi(n, p=0.1)
        >>> model = RandomGraph(n)

        Homogeneous (Erdős-Rényi) case.
        >>> fit = model.fit(G, "homogeneous")
        >>> H = fit.hamiltonian()
        >>> H_naive = fit.model.mu.theta * G.ecount()
        >>> jnp.allclose(H, H_naive).item()
        True

        Heterogeneous (soft configuration model) case.
        >>> fit = model.fit(G, "heterogeneous")
        >>> H = fit.hamiltonian()
        >>> H_naive = jnp.sum(fit.model.mu.theta * jnp.array(G.degree()))
        >>> jnp.allclose(H, H_naive).item()
        True
        """
        if model is None:
            model = self.model  # type: ignore[assignment]
        stats = self.sufficient_statistics
        if normalize:
            stats = stats.normalize(model.n_nodes)
        nodes = model.nodes
        H = 0.0
        for param in nodes.parameters:
            statname, _ = param.get_statistic(model, self.method)
            # 'param.theta' is Lagrange multiplier
            hvec = param.theta * stats[statname]
            H += jnp.sum(hvec)
        return H

    def lagrangian(
        self,
        model: "AbstractErgm | None" = None,
        *,
        normalize: bool = False,
    ) -> Real:
        """Compute the Lagrangian of the model.

        Parameters
        ----------
        model
            The model to compute the Lagrangian for.
            If `None`, uses the model associated with this fit.
        normalize
            Whether to normalize the Lagrangian by the number of nodes.

        Examples
        --------
        >>> import random
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import igraph as ig
        >>> from grgg import RandomGenerator, RandomGraph
        >>> random.seed(17)
        >>> rng = RandomGenerator(0)
        >>> n = 1000
        >>> G = ig.Graph.Erdos_Renyi(n, p=10/n)
        >>> def naive_lagrangian(model, G, normalize=False):
        ...     if model.is_homogeneous:
        ...         H = model.mu.theta * G.ecount()
        ...     else:
        ...         H = jnp.sum(model.mu.theta * jnp.array(G.degree()))
        ...     F = model.free_energy()
        ...     if normalize:
        ...         H /= model.n_nodes
        ...         F /= model.n_nodes
        ...     return H - F

        Homogeneous (Erdős-Rényi) case.
        >>> fit = RandomGraph(n).fit(G, mu=0)
        >>> jnp.isclose(fit.lagrangian(), naive_lagrangian(fit.model, G)).item()
        True
        >>> # Lagrangian objective is normalized by default
        >>> # See next section of this docstring for details.
        >>> objective = fit.define_objective(normalize=False)
        >>> grad = jax.grad(objective)(fit.model)
        >>> grad_naive = jax.grad(naive_lagrangian)(fit.model, G)
        >>> g, g_naive = grad.mu.data, grad_naive.mu.data
        >>> jnp.allclose(g, g_naive, atol=1e-3, rtol=1e-3).item()
        True

        Heterogeneous (soft configuration model) case.
        >>> fit = RandomGraph(n).fit(G, mu="zeros")
        >>> jnp.isclose(fit.lagrangian(), naive_lagrangian(fit.model, G)).item()
        True
        >>> objective = fit.define_objective(normalize=False)
        >>> grad = jax.grad(objective)(fit.model)
        >>> grad_naive = jax.grad(naive_lagrangian)(fit.model, G)
        >>> g, g_naive = grad.mu.data, grad_naive.mu.data
        >>> jnp.allclose(g, g_naive, atol=1e-3, rtol=1e-2).item()
        True

        Lagrangians (and Hamiltonians) can be normalized by the number of nodes.
        This is useful, for instance, for preventing overflows for computations on
        very large graphs (e.g. during parameter optimization).

        Homogeneous case with normalization.
        >>> fit = RandomGraph(n).fit(G, mu=0)
        >>> L = fit.lagrangian(normalize=True)
        >>> L_naive = naive_lagrangian(fit.model, G, normalize=True)
        >>> jnp.isclose(L, L_naive).item()
        True
        >>> objective = fit.define_objective(normalize=True)
        >>> grad = jax.grad(objective)(fit.model)
        >>> grad_naive = eqx.filter_grad(naive_lagrangian)(fit.model, G, normalize=True)
        >>> g, g_naive = grad.mu.data, grad_naive.mu.data
        >>> jnp.allclose(g, g_naive, atol=1e-3, rtol=1e-3).item()
        True
        """
        if model is None:
            model = self.model  # type: ignore[assignment]
        H = self.hamiltonian(model, normalize=normalize)
        F = model.free_energy(normalize=normalize)
        return H - F

    def define_objective(
        self,
        *,
        normalize: bool = True,
        options: Mapping[str, Any] | None = None,
        **stats_options: Mapping[str, Any],
    ) -> Real:
        """Define Lagrangian objective function."""

        stats = self.target.normalize(self.model.n_nodes) if normalize else self.target
        stats = stats.select(*self.get_expected_statistics_names())

        @eqx.filter_custom_vjp
        @eqx.filter_jit
        def objective(model: T, args: PyTree[Any] = None) -> Real:  # noqa
            """Lagrangian objective function.

            Parameters
            ----------
            model
                The model to evaluate the objective for.
            args
                Additional arguments (not used).
                Here for compatibility with :mod:`optimistix`.
            """
            return self.lagrangian(model, normalize=normalize)

        @objective.def_fwd
        @eqx.filter_jit
        def objective_fwd(_, model: T, args: PyTree[Any] = None) -> tuple[Real, None]:
            return objective(model, args), None

        @objective.def_bwd
        @eqx.filter_jit
        def objective_bwd(_, g_out: Real, __, model: T, args: PyTree[Any] = None) -> T:  # noqa
            expectations = self.compute_expectations(
                model, normalize=normalize, options=options, **stats_options
            )
            gradient = model.parameters.__class__(
                *jax.tree.map(lambda e, s: (e - s) * g_out, expectations, stats)
            )
            return model.replace(parameters=gradient)

        return objective
