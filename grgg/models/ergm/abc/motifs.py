from typing import TYPE_CHECKING, TypeVar

import equinox as eqx

from grgg.abc import AbstractModule
from grgg.statistics.motifs import (
    QHeadMotifStatistic,
    QuadrangleMotifStatistic,
    QWedgeMotifStatistic,
    THeadMotifStatistic,
    TriangleMotifStatistic,
    TWedgeMotifStatistic,
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

    def equals(self, other: object) -> bool:
        return super().equals(other) and self.view.equals(other.view)


class AbstractErgmNodeMotifs[V](AbstractErgmMotifs[V]):
    """Abstract base class for ERGM node motif statistics."""

    view: eqx.AbstractVar[V]

    @property
    def nodes(self) -> V:
        """The node view the motifs are computed for."""
        return self.view

    @property
    def triangle(self) -> TriangleMotifStatistic:
        """Triangle motif statistic for the nodes in the view."""
        return TriangleMotifStatistic.from_module(self)

    @property
    def twedge(self) -> TWedgeMotifStatistic:
        """Triangle wedge motif statistic for the nodes in the view."""
        return TWedgeMotifStatistic.from_module(self)

    @property
    def thead(self) -> THeadMotifStatistic:
        """Triangle head motif statistic for the nodes in the view."""
        return THeadMotifStatistic.from_module(self)

    @property
    def quadrangle(self) -> QuadrangleMotifStatistic:
        """Quadrangle motif statistic for the nodes in the view."""
        return QuadrangleMotifStatistic.from_module(self)

    @property
    def qwedge(self) -> QWedgeMotifStatistic:
        """Quadrangle wedge motif statistic for the nodes in the view."""
        return QWedgeMotifStatistic.from_module(self)

    @property
    def qhead(self) -> QHeadMotifStatistic:
        """Quadrangle head motif statistic for the nodes in the view."""
        return QHeadMotifStatistic.from_module(self)


class AbstractErgmNodePairMotifs[E](AbstractErgmMotifs[E]):
    """Abstract base class for ERGM node pair motif statistics."""

    view: eqx.AbstractVar[E]

    @property
    def pairs(self) -> E:
        """The node pair view the motifs are computed for."""
        return self.view
