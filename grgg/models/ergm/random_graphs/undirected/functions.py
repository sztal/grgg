from typing import TYPE_CHECKING

import equinox as eqx

from grgg._typing import Reals

if TYPE_CHECKING:
    from .model import AbstractRandomGraph

__all__ = ("couplings",)


@eqx.filter_jit
def couplings(model: "AbstractRandomGraph", mu: Reals) -> Reals:  # noqa
    """Compute edge couplings."""
    return -mu


# @eqx.filter_jit
# def free_energy(model: "AbstractRandomGraph", mu: Reals) -> Real:
#     """Compute the free energy of the model."""
#     @fori(1, model.n_nodes, init=0.0)
#     def F(carry: Real, i: Integer) -> Real:
#         carry += model.pairs[i, :i].probs(log=True)
