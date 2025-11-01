from typing import TYPE_CHECKING, Any, TypeVar

import equinox as eqx

from grgg._typing import Reals

from ..abc import AbstractRandomGraphFunctions

if TYPE_CHECKING:
    from .model import AbstractRandomGraph

    T = TypeVar("T", bound=AbstractRandomGraph)

__all__ = ("RandomGraphFunctions",)


class RandomGraphFunctions[T](AbstractRandomGraphFunctions[T]):
    """Random graph model functions."""

    def couplings(self, mu: Reals) -> Reals:
        """Compute edge couplings."""
        return couplings(self.model, mu)

    def free_energy(self, *args: Any, **kwargs: Any) -> Reals:
        return super().free_energy(*args, **kwargs) / 2


@eqx.filter_jit
def couplings(funcs: RandomGraphFunctions, mu: Reals) -> Reals:  # noqa
    """Compute edge couplings."""
    return -mu
