from typing import TYPE_CHECKING

from grgg._typing import Reals
from grgg.models.base.random_graphs import AbstractRandomGraphFunctions

if TYPE_CHECKING:
    from .model import RandomGraph

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
    def couplings(cls, params: "RandomGraph.Parameters.Data") -> Reals:
        """Compute edge couplings."""
        return params.mu.theta
