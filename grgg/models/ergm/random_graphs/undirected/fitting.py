from collections.abc import Mapping
from typing import Any

import equinox as eqx
import jax.numpy as jnp

from grgg._typing import Reals
from grgg.models.abc import AbstractModelFit
from grgg.models.ergm.abc import AbstractSufficientStatistics, LagrangianErgmFit

from ..abc.parameters import Mu

__all__ = ("RandomGraphSufficientStatistics",)


class RandomGraphSufficientStatistics(AbstractSufficientStatistics):
    """Sufficient statistics for random graph ERGM models.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from grgg import RandomGraph
    >>> n = 100
    >>> model = RandomGraph(n)
    >>> A = jnp.ones((n, n)) - jnp.eye(n)
    >>> fit = model.fit(A)
    >>> fit.target
    RandomGraphSufficientStatistics(degree=9900.00)
    >>> fit = model.fit(A, heterogeneous=True)
    >>> fit.target
    RandomGraphSufficientStatistics(degree=f...[100])
    >>> fit = model.fit(A, homogeneous=False)
    >>> fit.target
    RandomGraphSufficientStatistics(degree=f...[100])
    >>> model = RandomGraph(n, mu=jnp.linspace(-3, 3, n))
    >>> fit = model.fit(A)
    >>> fit.target
    RandomGraphSufficientStatistics(degree=f...[100])
    >>> fit = model.fit(A, homogeneous=True)
    >>> fit.target
    RandomGraphSufficientStatistics(degree=9900.00)

    Determination of fit target is idempotent.
    >>> fit2 = model.fit(fit.target)
    >>> fit2.equals(fit)
    True

    Creation from explicit statistics.
    >>> model.fit(degree=10).target
    RandomGraphSufficientStatistics(degree=10)

    Check hamiltonian calculations.
    >>> model = RandomGraph(n)
    >>> fit = model.fit(A)
    >>> fit.hamiltonian().item()
    0.0
    >>> model = RandomGraph(n, mu=-2)
    >>> fit = model.fit(A)
    >>> fit.hamiltonian().item()
    9900.0
    >>> fit.hamiltonian().item() == model.hamiltonian(A).item()
    True

    Check lagrangian calculations.
    >>> fit.lagrangian().item() == model.lagrangian(A).item()
    True
    """

    degree: Reals = eqx.field(converter=jnp.asarray)

    @classmethod
    def get_stats2params(cls) -> Mapping[str, str]:
        return {"degree": "mu"}


@AbstractModelFit._make_target.dispatch
def _make_target(
    fitter_cls: type[LagrangianErgmFit],  # noqa
    model: "RandomGraph",
    data: Any,
) -> RandomGraphSufficientStatistics:
    """Create target statistics for fitting random graph based on model Lagrangian."""
    degree = model.nodes.degree.observed(data)
    return RandomGraphSufficientStatistics(degree=degree)


@AbstractModelFit._initialize_param.dispatch
def _initialize_param(
    model: "RandomGraph",  # noqa
    param: Mu,
    target: RandomGraphSufficientStatistics,
) -> Mu:
    D = target.degree
    return param.replace(data=jnp.log(D / jnp.sqrt(D.sum())))


# Handle circular imports ------------------------------------------------------------

from .model import RandomGraph  # noqa: E402
