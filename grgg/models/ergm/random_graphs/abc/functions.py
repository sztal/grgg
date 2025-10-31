from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax

from grgg._typing import Reals

if TYPE_CHECKING:
    from .model import AbstractRandomGraph

__all__ = ("logprobs", "probs")


@eqx.filter_jit
def logprobs(model: "AbstractRandomGraph", *args: Any, **kwargs: Any) -> Reals:
    """Compute edge log-probabilities."""
    couplings = model.couplings(*args, **kwargs)
    return jax.nn.log_sigmoid(-couplings)


@eqx.filter_jit
def probs(model: "AbstractRandomGraph", *args: Any, **kwargs: Any) -> Reals:
    """Compute edge probabilities."""
    couplings = model.couplings(*args, **kwargs)
    return jax.nn.sigmoid(-couplings)
