from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax.numpy as jnp

from grgg._typing import Reals
from grgg.models.abc import AbstractModelFunctions

if TYPE_CHECKING:
    from grgg.models.ergm.abc import AbstractErgmModel

__all__ = ("AbstractErgmFunctions",)


class AbstractErgmFunctions(AbstractModelFunctions):
    """ERGM model functions container."""

    @classmethod
    @eqx.filter_jit
    def free_energy(
        cls, model: "AbstractErgmModel", *args: Any, **kwargs: Any
    ) -> Reals:
        """Compute the free energy of the model."""
        raise NotImplementedError

    @classmethod
    @eqx.filter_jit
    def partition_function(
        cls, model: "AbstractErgmModel", *args: Any, **kwargs: Any
    ) -> Reals:
        """Compute the partition function of the model."""
        free_energy = cls.free_energy(model, *args, **kwargs)
        return jnp.exp(-free_energy)
