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
    def define_integrand(self) -> IntegrandT:
        """Define the integrand function."""

    def integrate(
        self,
        *args: Any,
        limits: tuple[float, float] | None = None,
        method: IntegratorT | None = None,
        breakpoints: Sequence[float] | None = None,
        pass_options: bool = False,
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
        pass_options
            If `True`, passes `**kwargs` options as a an extra last item in `args`
            to the integrand function (so it must accept it). This is useful for passing
            options to inner computations in multi-dimensional integrals.
        **kwargs
            Additional keyword arguments passed to the integrator.
        """
        method, interval, options = self.make_options(
            limits, method=method, breakpoints=breakpoints, **kwargs
        )
        integrand = eqx.filter_jit(self.define_integrand())
        if pass_options:
            args = (*args, options)
        integral, info = method(integrand, interval, args, **options)
        return self.constant * integral, info

    def make_options(
        self,
        limits: tuple[float, float] | None = None,
        method: IntegratorT | None = None,
        breakpoints: Sequence[float] | None = None,
        **kwargs: Any,
    ) -> tuple[IntegratorT, jnp.ndarray, dict[str, Any]]:
        """Create integrator options.

        Parameters
        ----------
        limits
            Integration limits as a tuple `(a, b)`. If `None`, uses `self.domain`.
        method
            Integration method to use. If `None`, uses `self.default_integrator`.
        breakpoints
            Breakpoints to focus the integration.
        **kwargs
            Additional keyword arguments passed to the integrator.

        Returns
        -------
        method
            The integration method to use.
        interval
            The integration interval with breakpoints included.
        options
            The integrator options.
        """
        if method is None:
            method = self.default_integrator
        if limits is None:
            limits = self.domain
        lo, up = limits
        interval = jnp.array([lo, *(breakpoints or ()), up])
        options = {**self.defaults, **kwargs}
        return method, interval, options
