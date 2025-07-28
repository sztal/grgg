from dataclasses import dataclass
from typing import Any, Self

import numpy as np

from .utils import (
    copy_with_update,
    sphere_radius,
    sphere_surface_area,
    sphere_surface_sample,
    sphere_volume,
)


@dataclass
class Sphere:
    """Sphere in :math:`k`-dimensional space.

    Attributes
    ----------
    R : float
        Radius of the sphere.
    k : int
        Surface dimension of the sphere.

    Examples
    --------
    >>> sphere = Sphere(R=1.0, k=2)
    >>> sphere.S  # doctest: +FLOAT_CMP
    12.566370614359172
    >>> sphere.V  # doctest: +FLOAT_CMP
    4.1887902047863905
    >>> sphere = Sphere.from_area(S=100.0, k=2)
    >>> sphere.S  # doctest: +FLOAT_CMP
    100.0
    """

    R: float
    k: int

    def __post_init__(self) -> None:
        """Validate the sphere's attributes."""
        if self.R <= 0:
            errmsg = "Radius must be positive."
            raise ValueError(errmsg)
        if self.k < 1:
            errmsg = "Surface dimension must be at least 1."
            raise ValueError(errmsg)
        self.R = float(self.R)
        self.k = int(self.k)

    def __copy__(self) -> Self:
        """Create a shallow copy of the sphere."""
        return self.__class__(self.R, self.k)

    @property
    def S(self) -> float:
        """Compute the surface area of the sphere."""
        return sphere_surface_area(self.R, self.k)

    @property
    def V(self) -> float:
        """Compute the volume of the sphere."""
        return sphere_volume(self.R, self.k)

    @classmethod
    def from_area(cls, S: float, k: int) -> Self:
        """Create a sphere instance from its surface area."""
        R = sphere_radius(S, k)
        return cls(R=R, k=k)

    def copy(self, **kwargs: Any) -> Self:
        """Create a copy of the sphere with optional modifications."""
        return copy_with_update(self, **kwargs)

    def sample_surface(self, n: int, **kwargs: Any) -> np.ndarray:
        """Sample points uniformly from the surface of the sphere."""
        return sphere_surface_sample(n, self.k, **kwargs) * self.R
