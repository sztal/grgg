from typing import ClassVar

import jax.numpy as jnp

from grgg.models.abc import Constraints
from grgg.models.ergm.abc import AbstractErgmParameter
from grgg.statistics.degree import Degree


class Mu(AbstractErgmParameter):
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

    default_value: ClassVar[float] = 0.0
    ndims: ClassVar[tuple[int, ...]] = (0, 1)
    constraints: ClassVar[tuple[Constraints, ...]] = (Constraints.REAL,)

    statistic: ClassVar[type[Degree]] = Degree

    @property
    def theta(self) -> jnp.ndarray:
        """Raw Lagrange multiplier representation of the parameter."""
        return -self.data
