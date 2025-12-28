from functools import wraps
from typing import TYPE_CHECKING, TypeVar

import equinox as eqx

from grgg.statistics.motifs import (
    AbstractErgmNodeMotifStatistic,
    QHeadMotif,
    QuadrangleMotif,
    QWedgeMotif,
    THeadMotif,
    TriangleMotif,
    TWedgeMotif,
)
from grgg.utils.dispatch import dispatch

from ..model import AbstractModelModule

if TYPE_CHECKING:
    from .model import AbstractErgm
    from .views import AbstractErgmNodePairView, AbstractErgmNodeView, AbstractErgmView

__all__ = ("AbstractErgmMotifs", "ErgmNodeMotifs", "ErgmNodePairMotifs")


T = TypeVar("T", bound="AbstractErgm")
V = TypeVar("V", bound="AbstractErgmView[T]")
NV = TypeVar("NV", bound="AbstractErgmNodeView[T]")
PV = TypeVar("PV", bound="AbstractErgmNodePairView[T]")


class AbstractErgmMotifs[T](AbstractModelModule[T]):
    """Abstract base class for ERGM motif statistics."""

    view: eqx.AbstractVar["V"]

    @property
    def model(self) -> T:
        """The model the motifs are computed for."""
        return self.view.model

    @dispatch.abstract
    def _get_motif(
        self,
        motif: str,
        view: T,
    ) -> AbstractErgmNodeMotifStatistic:
        """Get the motif statistic by name for the given model."""


class ErgmNodeMotifs[T](AbstractErgmMotifs[T]):
    """ERGM node motif statistics."""

    view: "NV"

    @property
    def nodes(self) -> "NV":
        """The node view the motifs are computed for."""
        return self.view

    @property
    @wraps(TriangleMotif.__init__)
    def triangle(self) -> TriangleMotif:
        """Triangle motif statistic for the nodes in the view."""
        return self._get_motif("triangle", self.view)

    @property
    @wraps(TWedgeMotif.__init__)
    def twedge(self) -> TWedgeMotif:
        """Triangle wedge motif statistic for the nodes in the view."""
        return self._get_motif("twedge", self.view)

    @property
    @wraps(THeadMotif.__init__)
    def thead(self) -> THeadMotif:
        """Triangle head motif statistic for the nodes in the view."""
        return self._get_motif("thead", self.view)

    @property
    @wraps(QuadrangleMotif.__init__)
    def quadrangle(self) -> QuadrangleMotif:
        """Quadrangle motif statistic for the nodes in the view."""
        return self._get_motif("quadrangle", self.view)

    @property
    @wraps(QWedgeMotif.__init__)
    def qwedge(self) -> QWedgeMotif:
        """Quadrangle wedge motif statistic for the nodes in the view."""
        return self._get_motif("qwedge", self.view)

    @property
    @wraps(QHeadMotif.__init__)
    def qhead(self) -> QHeadMotif:
        """Quadrangle head motif statistic for the nodes in the view."""
        return self._get_motif("qhead", self.view)


class ErgmNodePairMotifs[T](AbstractErgmMotifs[T]):
    """ERGM node pair motif statistics."""

    view: "PV"

    @property
    def pairs(self) -> "PV":
        """The node pair view the motifs are computed for."""
        return self.view
