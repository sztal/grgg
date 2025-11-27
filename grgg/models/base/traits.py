from typing import ClassVar

import equinox as eqx

__all__ = (
    "ModelTraitMixin",
    "EdgeDirection",
    "Undirected",
    "Directed",
    "EdgeWeighting",
    "Unweighted",
    "Weighted",
)


class ModelTraitMixin(eqx.Module):
    """Base class for model traits."""

    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()


class EdgeDirection(ModelTraitMixin):
    """Trait for edge directionality in network models."""

    is_directed: eqx.AbstractClassVar[bool]


class Undirected(EdgeDirection):
    """Trait indicating that the model is for undirected networks."""

    is_directed: ClassVar[bool] = False


class Directed(EdgeDirection):
    """Trait indicating that the model is for directed networks."""

    is_directed: ClassVar[bool] = True


class EdgeWeighting(ModelTraitMixin):
    """Trait for edge weighting in network models."""

    is_weighted: eqx.AbstractClassVar[bool]


class Unweighted(EdgeWeighting):
    """Trait indicating that the model is for unweighted networks."""

    is_weighted: ClassVar[bool] = False


class Weighted(EdgeWeighting):
    """Trait indicating that the model is for weighted networks."""

    is_weighted: ClassVar[bool] = True
