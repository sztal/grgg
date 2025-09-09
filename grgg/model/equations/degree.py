from collections.abc import Callable
from functools import singledispatchmethod

import numpy as np

from grgg.manifolds import CompactManifold, Sphere

from .abc import ModelEquation


class DegreeEquation(ModelEquation):
    """Equation for the average degree in the GRGG model.

    Attributes
    ----------
    manifold
        The manifold on which the equation is defined.
    """

    @singledispatchmethod
    def define_constant(self, manifold: CompactManifold) -> float:
        errmsg = f"Constant not implemented for '{manifold}'."
        raise NotImplementedError(errmsg)

    @define_constant.register
    def _(self, manifold: Sphere) -> float:  # noqa
        delta = self.model.delta
        d = self.manifold.dim
        R = self.manifold.radius
        dV = Sphere(d - 1).volume
        return delta * dV * R**d


class HomogeneousDegreeEquation(DegreeEquation):
    """Equation for the average degree in the homogeneous GRGG model.

    Attributes
    ----------
    manifold
        The manifold on which the equation is defined.
    """

    @singledispatchmethod
    def define_affine(
        self, manifold: CompactManifold
    ) -> Callable[[np.ndarray], np.ndarray]:
        errmsg = f"Affine not implemented for '{manifold}'."
        raise NotImplementedError(errmsg)

    @define_affine.register
    def _(self, manifold: Sphere) -> Callable[[np.ndarray], np.ndarray]:  # noqa
        def affine(self, x: np.ndarray) -> np.ndarray:
            C = (1 - 1 / self.model.n_nodes) * self.constant
            return C * x

        return affine

    @singledispatchmethod
    def define_function(
        self, manifold: CompactManifold
    ) -> Callable[[np.ndarray], np.ndarray]:
        errmsg = f"Function not implemented for '{manifold}'."
        raise NotImplementedError(errmsg)

    @define_function.register
    def _(self, manifold: Sphere) -> Callable[[np.ndarray], np.ndarray]:  # noqa
        pass
