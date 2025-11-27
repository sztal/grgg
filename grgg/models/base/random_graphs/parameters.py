from typing import ClassVar

import jax.numpy as jnp

from grgg.models.base.model import AbstractParameter
from grgg.utils.variables import Constraints


class Mu(AbstractParameter):
    """Mu parameter (chemical potential).

    It controls the density of the network.

    Attributes
    ----------
    value
        Parameter value.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> mu = Mu()
    >>> mu.data  # default value
    Array(0.0, ...)
    >>> Mu(1.0)  # homogeneous value
    Mu(1.0)
    >>> Mu([1, 2, 3])  # heterogeneous value
    Mu(f32[3])
    """

    constraints: ClassVar[Constraints] = Constraints("real")
    default_value: ClassVar[float] = 0.0

    @property
    def theta(self) -> jnp.ndarray:
        """Raw Lagrange multiplier representation of the parameter."""
        return -self.data
