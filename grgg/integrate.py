from abc import abstractmethod
from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import quadax
from quadax.utils import QuadratureInfo

IntegrandT = Callable[[jnp.ndarray, ...], jnp.ndarray]
IntegrationResultT = tuple[jnp.ndarray, QuadratureInfo]
IntegratorT = Callable[[IntegrandT, tuple[float, float], ...], IntegrationResultT]


class AbstractIntegral(eqx.Module):
    """Abstract base class for integral computations."""

    def __call__(self, x: jnp.ndarray, *args: Any, **kwargs: Any) -> jnp.ndarray:
        """Evaluate the integral at given points `x`."""
        return self.constant * self.integrand(x, *args, **kwargs)

    @property
    @abstractmethod
    def domain(self) -> tuple[float, float]:
        """Return the leading integration domain as a tuple `(a, b)`."""
        return (-jnp.inf, jnp.inf)

    @property
    def breakpoints(self) -> Sequence[float]:
        """Default breakpoints for integration focus."""
        return []

    @property
    @abstractmethod
    def constant(self) -> float:
        """Return the constant multiplier for the integral."""
        return 1.0

    @property
    @abstractmethod
    def defaults(self) -> dict[str, Any]:
        """Return default parameters for the integral."""
        return {}

    @property
    def default_integrator(self) -> IntegratorT:
        """Default integrator function."""
        return quadax.quadcc

    @abstractmethod
    def integrand(self, x: jnp.ndarray, *args: Any, **kwargs: Any) -> jnp.ndarray:
        """Compute the integrand at given points `x`."""

    def integrate(
        self,
        *args: Any,
        limits: tuple[float, float] | None = None,
        method: IntegratorT | None = None,
        breakpoints: Sequence[float] | None = None,
        **kwargs: Any,
    ) -> IntegrationResultT:
        """Compute the integral.

        Parameters
        ----------
        *args
            Additional positional arguments passed to the integrand.
        limits
            Integration limits as a tuple `(a, b)`. If `None`, uses `self.domain`.
        method
            Integration method to use. If `None`, uses `self.default_integrator`.
        breakpoints
            Breakpoints to focus the integration. If `None`, uses `self.breakpoints`.
        **kwargs
            Additional keyword arguments passed to the integrator.
        """
        if limits is None:
            limits = self.domain
        if method is None:
            method = self.default_integrator
        if breakpoints is None:
            breakpoints = self.breakpoints
        interval = jnp.array([limits[0], *breakpoints, limits[1]])
        options = {**self.defaults, **kwargs}
        integrand = eqx.filter_jit(self.integrand)
        integral, info = method(integrand, interval, args, **options)
        return self.constant * integral, info
