from functools import wraps
from typing import TYPE_CHECKING, TypeVar

import equinox as eqx

from grgg.models.abc import AbstractModelModule
from grgg.statistics.motifs import (
    QHeadMotif,
    QuadrangleMotif,
    QWedgeMotif,
    THeadMotif,
    TriangleMotif,
    TWedgeMotif,
)

if TYPE_CHECKING:
    from .model import AbstractErgm
    from .views import AbstractErgmNodePairView, AbstractErgmNodeView, AbstractErgmView

    T = TypeVar("T", bound=AbstractErgm)
    V = TypeVar("V", bound=AbstractErgmView[T])
    NV = TypeVar("NV", bound=AbstractErgmNodeView[T])
    PV = TypeVar("PV", bound=AbstractErgmNodePairView[T])

__all__ = ("AbstractErgmMotifs", "AbstractErgmNodeMotifs", "AbstractErgmNodePairMotifs")


class AbstractErgmMotifs[T](AbstractModelModule[T]):
    """Abstract base class for ERGM motif statistics."""

    view: eqx.AbstractVar["V"]

    @property
    def model(self) -> T:
        """The model the motifs are computed for."""
        return self.view.model

    def _equals(self, other: object) -> bool:
        return super()._equals(other) and self.view.equals(other.view)


class AbstractErgmNodeMotifs[T](AbstractErgmMotifs[T]):
    """Abstract base class for ERGM node motif statistics."""

    view: eqx.AbstractVar["NV"]

    triangle_cls: eqx.AbstractClassVar[type[TriangleMotif]]
    twedge_cls: eqx.AbstractClassVar[type[TWedgeMotif]]
    thead_cls: eqx.AbstractClassVar[type[THeadMotif]]
    quadrangle_cls: eqx.AbstractClassVar[type[QuadrangleMotif]]
    qwedge_cls: eqx.AbstractClassVar[type[QWedgeMotif]]
    qhead_cls: eqx.AbstractClassVar[type[QHeadMotif]]

    @property
    def nodes(self) -> "NV":
        """The node view the motifs are computed for."""
        return self.view

    @property
    @wraps(TriangleMotif.__init__)
    def triangle(self) -> TriangleMotif:
        """Triangle motif statistic for the nodes in the view."""
        return self.triangle_cls(self)

    @property
    @wraps(TWedgeMotif.__init__)
    def twedge(self) -> TWedgeMotif:
        """Triangle wedge motif statistic for the nodes in the view."""
        return self.twedge_cls(self)

    @property
    @wraps(THeadMotif.__init__)
    def thead(self) -> THeadMotif:
        """Triangle head motif statistic for the nodes in the view."""
        return self.thead_cls(self)

    @property
    @wraps(QuadrangleMotif.__init__)
    def quadrangle(self) -> QuadrangleMotif:
        """Quadrangle motif statistic for the nodes in the view."""
        return self.quadrangle_cls(self)

    @property
    @wraps(QWedgeMotif.__init__)
    def qwedge(self) -> QWedgeMotif:
        """Quadrangle wedge motif statistic for the nodes in the view."""
        return self.qwedge_cls(self)

    @property
    @wraps(QHeadMotif.__init__)
    def qhead(self) -> QHeadMotif:
        """Quadrangle head motif statistic for the nodes in the view."""
        return self.qhead_cls(self)


class AbstractErgmNodePairMotifs[T](AbstractErgmMotifs[T]):
    """Abstract base class for ERGM node pair motif statistics."""

    view: eqx.AbstractVar["PV"]

    @property
    def pairs(self) -> "PV":
        """The node pair view the motifs are computed for."""
        return self.view
