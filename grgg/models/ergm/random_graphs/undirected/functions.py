from typing import TYPE_CHECKING, Any

import equinox as eqx

from grgg._typing import Integer, Real, Reals

from ..abc import AbstractRandomGraphFunctions

if TYPE_CHECKING:
    from .model import RandomGraph
    from .views import RandomGraphNodeView

__all__ = ("RandomGraphFunctions",)


class RandomGraphFunctions(AbstractRandomGraphFunctions):
    """Random graph model functions.

    Examples
    --------
    >>> import jax
    >>> from grgg import RandomGraph, RandomGenerator
    >>> n = 100
    >>> rng = RandomGenerator(303)
    >>> model = RandomGraph(n, mu=rng.normal(n) - 2.5)
    >>> fe0 = model.pairs.free_energy().sum() / 2
    >>> fe1 = model.free_energy()
    >>> jax.numpy.isclose(fe0, fe1).item()
    True
    """

    @classmethod
    @eqx.filter_jit
    def couplings(cls, params: "RandomGraph.Parameters.Data") -> Reals:
        """Compute edge couplings."""
        return -params.mu

    @classmethod
    @eqx.filter_jit
    def F_i(cls, model: "RandomGraph", i: Integer, *args: Any, **kwargs: Any) -> Real:
        """Compute the contribution to the free energy from node i."""
        return super().F_i(model, i, *args, **kwargs) / 2

    @classmethod
    @eqx.filter_jit
    def _node_free_energy_homogeneous(
        cls, nodes: "RandomGraphNodeView", *args: Any, **kwargs: Any
    ) -> Reals:
        """Compute the free energy contributions from nodes in a homogeneous model."""
        return super()._node_free_energy_homogeneous(nodes, *args, **kwargs) / 2
