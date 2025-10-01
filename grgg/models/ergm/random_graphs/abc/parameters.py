from typing import ClassVar

from grgg.models.abc import AbstractParameter, Constraints


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

    Error is raised for invalid values.
    >>> Mu(1+1j)  # complex value
    Traceback (most recent call last):
        ...
    ValueError: 'mu' must be real

    >>> Mu(jnp.ones((2, 2)))  # wrong shape
    Traceback (most recent call last):
        ...
    ValueError: 'data.ndim' must be in (0, 1), got 2
    """

    default_value: ClassVar[float] = 0.0
    ndims: ClassVar[tuple[int, ...]] = (0, 1)
    constraints: ClassVar[tuple[Constraints, ...]] = (Constraints.REAL,)
