import math
from collections.abc import Mapping
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from grgg.integrate import IntegrandT

from .abc import AbstractNodesIntegral

__all__ = ("TClusteringIntegral",)


class TClusteringIntegral(AbstractNodesIntegral):
    """Triangle clustering integral.

    Attributes
    ----------
    nodes
        The nodes the integral is defined on.
    inner_opts
        Integration options for the inner integrals.
    inner_method
        Integrator for the inner integrals.
        Defaults to `self.default_integrator`.
    unit_subspace
        A submanifold in one lower dimension with unit linear size.

    Notes
    -----
    The main integration variable is the angle `theta` formed by the tangent vectors
    of geodesics connecting the central node to two other nodes.
    """

    _inner_opts: tuple[tuple[str, Any], ...] = eqx.field(static=True, repr=False)

    def __init__(
        self,
        *args: Any,
        inner_opts: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._inner_opts = tuple(inner_opts.items()) if inner_opts else ()

    @property
    def constant(self) -> float:
        d = self.manifold.dim
        pbar = self.model.pairs.expected_probs()
        if d == 1:
            num = math.gamma((d + 1) / 2) ** 2
            den = 2 * pbar**2 * math.pi * math.gamma(d / 2) ** 2
        else:
            num = math.gamma((d + 1) / 2) ** 2
            den = (
                pbar**2
                * math.pi ** (3 / 2)
                * math.gamma(d / 2)
                * math.gamma((d - 1) / 2)
            )
        return num / den

    @property
    def inner_opts(self) -> dict[str, Any]:
        return dict(self._inner_opts)


@TClusteringIntegral.register_homogeneous
class HomogeneousTClusteringIntegral(TClusteringIntegral):
    """Homogeneous triangle clustering integral.

    Attributes
    ----------
    model
        The model the integral is defined on.

    Notes
    -----
    The integration is always done on the unit sphere, and only rescaled to the manifold
    radius in the constant multiplier. Thus, the domain of integration is `[0, pi]

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from grgg import GRGG, Sphere, Similarity, Complementarity, RandomGenerator
    >>> from grgg.model.integrals import TClusteringIntegral
    >>> rng = RandomGenerator(42)
    >>> model = GRGG(100, 2, Similarity)
    >>> integral = TClusteringIntegral.from_model(model)
    >>> tclust = integral.integrate()[0]
    >>> tclust.item()
    0.24052459

    >>> def sample_tclust(n: int) -> jnp.ndarray:
    ...     samples = []
    ...     for _ in range(n):
    ...         G = model.sample(rng=rng).G
    ...         tclust = G.transitivity_local_undirected()
    ...         samples.append(jnp.nanmean(jnp.asarray(tclust)))
    ...     return jnp.array(samples)
    >>>
    >>> C = sample_tclust(50)
    >>> jnp.isclose(tclust, C.mean(), atol=1e-2, rtol=1e-2).item()
    True

    Check that it works for non-standard volume and dimensions.
    >>> model = GRGG(100, Sphere(4, 2.0), Similarity, Complementarity)
    >>> integral = TClusteringIntegral.from_model(model)
    >>> tclust = integral.integrate()[0]
    >>> tclust.item()
    0.19198909
    >>> C = sample_tclust(100)
    >>> jnp.isclose(tclust, C.mean(), atol=1e-2, rtol=1e-2).item()
    True

    Check special `d=1` case.
    >>> model = GRGG(100, 1, Similarity)
    >>> integral = TClusteringIntegral.from_model(model)
    >>> tclust = integral.integrate()[0]
    >>> tclust.item()
    0.22321113
    >>> C = sample_tclust(100)
    >>> jnp.isclose(tclust, C.mean(), atol=1e-2, rtol=1e-2).item()
    True
    """

    def define_integrand(self) -> IntegrandT:
        @jax.jit
        def inner_theta_k(
            theta_k: jnp.ndarray,
            g_ij: jnp.ndarray,
            g_ik: jnp.ndarray,
        ) -> jnp.ndarray:
            g_jk = self.manifold.cosine_law(theta_k, g_ij, g_ik)
            p_jk = self.model.pairs.probs(g_jk)
            return p_jk * jnp.sin(theta_k) ** (self.manifold.dim - 2)

        @jax.jit
        def inner_phi_k(
            phi_k: jnp.ndarray, g_ij: jnp.ndarray, *args: Any
        ) -> jnp.ndarray:
            options = args[0] if args else {}
            options = {**self.inner_opts, **options}
            g_ik = phi_k * self.manifold.linear_size
            p_ik = self.model.pairs.probs(g_ik)
            if self.manifold.dim == 1:
                g_jk = jnp.array([jnp.abs(g_ij - g_ik), g_ij + g_ik])
                I_theta_k = self.model.pairs.probs(g_jk).sum()
            else:
                method, interval, options = self.make_options(**options)
                I_theta_k, _ = method(inner_theta_k, interval, (g_ij, g_ik), **options)
            return I_theta_k * p_ik * jnp.sin(phi_k) ** (self.manifold.dim - 1)

        @jax.jit
        def integrand(phi_j: jnp.ndarray, *args: Any) -> jnp.ndarray:
            options = args[0] if args else {}
            options = {**self.inner_opts, **options}
            g_ij = phi_j * self.manifold.linear_size
            p_ij = self.model.pairs.probs(g_ij)
            method, interval, options = self.make_options(**options)
            I_phi_k, _ = method(inner_phi_k, interval, (g_ij,), **options)
            return I_phi_k * p_ij * jnp.sin(phi_j) ** (self.manifold.dim - 1)

        return integrand
