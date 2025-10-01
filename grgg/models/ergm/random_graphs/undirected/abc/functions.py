from grgg._typing import Reals
from grgg.models.ergm.random_graphs.abc import AbstractCoupling

__all__ = ("UndirectedRandomGraphCoupling",)


class UndirectedRandomGraphCoupling(AbstractCoupling):
    """Abstract base class for undirected coupling functions."""

    def __call__(self, mu: Reals) -> Reals:
        """Evaluate the coupling function."""
        return -mu
