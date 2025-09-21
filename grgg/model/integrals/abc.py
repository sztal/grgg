from abc import abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Self

import jax.numpy as jnp

from grgg.abc import AbstractGRGG, AbstractModule
from grgg.integrate import AbstractIntegral

if TYPE_CHECKING:
    from grgg.model._views import NodePairView, NodeView
    from grgg.model.grgg import GRGG


class AbstractModelIntegral(AbstractIntegral, AbstractModule):
    """Abstract base class for model integrals.

    It is also used to construct concrete model integrals using
    :meth:`from_model`.

    Attributes
    ----------
    model
        The model the integral is defined on.

    Notes
    -----
    The integration is always done on the unit sphere, and only rescaled to the manifold
    radius in the constant multiplier. Thus, the domain of integration is `[0, pi]`.
    """

    homogeneous_type: ClassVar[type[Self]]
    heterogeneous_type: ClassVar[type[Self]]

    @property
    @abstractmethod
    def model(self) -> "GRGG":
        """The model the integral is defined on."""

    @property
    def domain(self) -> tuple[float, float]:
        return (0, jnp.pi)

    @property
    def constant(self) -> float:
        delta = self.model.delta
        R = self.model.manifold.linear_size
        d = self.model.manifold.dim
        dV = self.model.manifold.__class__(d - 1).volume
        return delta * R**d * dV

    @property
    def defaults(self) -> dict[str, Any]:
        return super().defaults

    def equals(self, other: object) -> bool:
        if not super().equals(other):
            return False
        dct1 = self.__dict__.copy()
        mod1 = dct1.pop("model", None)
        dct2 = other.__dict__.copy()
        mod2 = dct2.pop("model", None)
        return mod1.equals(mod2) and dct1 == dct2

    @classmethod
    @abstractmethod
    def from_model(cls, model: "GRGG", *args: Any, **kwargs: Any) -> Self:
        """Construct a model integral from a model."""

    @classmethod
    def register_homogeneous(cls, integral_type: type[Self]) -> None:
        """Register a concrete homogeneous model integral type."""
        if not issubclass(integral_type, cls):
            errmsg = f"integral_type must be a subclass of '{cls.__name__}'"
            raise TypeError(errmsg)
        cls.homogeneous_type = integral_type
        return integral_type

    @classmethod
    def register_heterogeneous(cls, integral_type: type[Self]) -> None:
        """Register a concrete heterogeneous model integral type."""
        if not issubclass(integral_type, cls):
            errmsg = f"integral_type must be a subclass of '{cls.__name__}'"
            raise TypeError(errmsg)
        cls.heterogeneous_type = integral_type
        return integral_type


class AbstractNodesIntegral(AbstractModelIntegral):
    """Abstract base class for node-related model integrals.

    Attributes
    ----------
    model
        The model the integral is defined on.

    Notes
    -----
    The integration is always done on the unit sphere, and only rescaled to the manifold
    radius in the constant multiplier. Thus, the domain of integration is `[0, pi]`.
    """

    nodes: "NodeView"

    @property
    def model(self) -> "GRGG":
        return self.nodes.module

    @classmethod
    def from_nodes(cls, nodes: "NodeView", *args: Any, **kwargs: Any) -> Self:
        """Construct a model integral from a model."""
        if not isinstance(nodes.module, AbstractGRGG):
            errmsg = (
                f"model must be an instance of AbstractGRGG, "
                f"got '{type(nodes.module).__name__}'"
            )
            raise TypeError(errmsg)
        if nodes.module.is_heterogeneous:
            return cls.heterogeneous_type(nodes, *args, **kwargs)
        return cls.homogeneous_type(nodes, *args, **kwargs)

    @classmethod
    def from_model(cls, model: "GRGG", *args: Any, **kwargs: Any) -> Self:
        """Construct a model integral from a model."""
        return cls.from_nodes(model.nodes, *args, **kwargs)


class AbstractPairsIntegral(AbstractModelIntegral):
    """Abstract base class for node-pair-related model integrals.

    Attributes
    ----------
    model
        The model the integral is defined on.

    Notes
    -----
    The integration is always done on the unit sphere, and only rescaled to the manifold
    radius in the constant multiplier. Thus, the domain of integration is `[0, pi]`.
    """

    pairs: "NodePairView"

    @property
    def model(self) -> "GRGG":
        return self.pairs.module

    @classmethod
    def from_pairs(cls, pairs: "NodePairView", *args: Any, **kwargs: Any) -> Self:
        """Construct a model integral from a model."""
        if not isinstance(pairs.module, AbstractGRGG):
            errmsg = (
                f"model must be an instance of AbstractGRGG, "
                f"got '{type(pairs.module).__name__}'"
            )
            raise TypeError(errmsg)
        if pairs.module.is_heterogeneous:
            return cls.heterogeneous_type(pairs, *args, **kwargs)
        return cls.homogeneous_type(pairs, *args, **kwargs)

    @classmethod
    def from_model(cls, model: "GRGG", *args: Any, **kwargs) -> Self:
        """Construct a model integral from a model."""
        return cls.from_pairs(model.pairs, *args, **kwargs)
