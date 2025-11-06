from typing import TYPE_CHECKING, Any

import equinox as eqx

from grgg._typing import Reals

from ..abc import AbstractRandomGraphFunctions

if TYPE_CHECKING:
    from .model import RandomGraph

__all__ = ("RandomGraphFunctions",)


class RandomGraphFunctions(AbstractRandomGraphFunctions):
    """Random graph model functions."""

    @classmethod
    @eqx.filter_jit
    def couplings(cls, params: "RandomGraph.Parameters") -> Reals:
        """Compute edge couplings."""
        return -params.mu

    @classmethod
    @eqx.filter_jit
    def free_energy(cls, model: "RandomGraph", *args: Any, **kwargs: Any) -> Reals:
        """Compute the free energy of the model.

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
        return super().free_energy(model, *args, **kwargs) / 2
