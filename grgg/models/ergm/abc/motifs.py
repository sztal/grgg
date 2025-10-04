from typing import TYPE_CHECKING, TypeVar

import equinox as eqx

from grgg.abc import AbstractModule
from grgg.statistics.motifs import (
    QHeadMotif,
    QuadrangleMotif,
    QWedgeMotif,
    THeadMotif,
    TriangleMotif,
    TWedgeMotif,
)

if TYPE_CHECKING:
    from .models import AbstractErgm
    from .views import AbstractErgmNodePairView, AbstractErgmNodeView, AbstractErgmView

    T = TypeVar("T", bound="AbstractErgm")
    U = TypeVar("U", bound=AbstractErgmView[T])
    E = TypeVar("E", bound=AbstractErgmNodePairView[T])
    V = TypeVar("V", bound=AbstractErgmNodeView[T])

__all__ = ("AbstractErgmMotifs", "AbstractErgmNodeMotifs", "AbstractErgmNodePairMotifs")


class AbstractErgmMotifs[U](AbstractModule):
    """Abstract base class for ERGM motif statistics."""

    view: eqx.AbstractVar[U]

    @property
    def model(self) -> "T":
        """The model the motifs are computed for."""
        return self.view.model

    def _equals(self, other: object) -> bool:
        return super()._equals(other) and self.view.equals(other.view)


class AbstractErgmNodeMotifs[V](AbstractErgmMotifs[V]):
    """Abstract base class for ERGM node motif statistics."""

    view: eqx.AbstractVar[V]

    @property
    def nodes(self) -> V:
        """The node view the motifs are computed for."""
        return self.view

    @property
    def triangle(self) -> TriangleMotif:
        """Triangle motif statistic for the nodes in the view."""
        return TriangleMotif.from_module(self)

    @property
    def twedge(self) -> TWedgeMotif:
        """Triangle wedge motif statistic for the nodes in the view."""
        return TWedgeMotif.from_module(self)

    @property
    def thead(self) -> THeadMotif:
        """Triangle head motif statistic for the nodes in the view."""
        return THeadMotif.from_module(self)

    @property
    def quadrangle(self) -> QuadrangleMotif:
        """Quadrangle motif statistic for the nodes in the view."""
        return QuadrangleMotif.from_module(self)

    @property
    def qwedge(self) -> QWedgeMotif:
        """Quadrangle wedge motif statistic for the nodes in the view."""
        return QWedgeMotif.from_module(self)

    @property
    def qhead(self) -> QHeadMotif:
        """Quadrangle head motif statistic for the nodes in the view."""
        return QHeadMotif.from_module(self)


class AbstractErgmNodePairMotifs[E](AbstractErgmMotifs[E]):
    """Abstract base class for ERGM node pair motif statistics."""

    view: eqx.AbstractVar[E]

    @property
    def pairs(self) -> E:
        """The node pair view the motifs are computed for."""
        return self.view
