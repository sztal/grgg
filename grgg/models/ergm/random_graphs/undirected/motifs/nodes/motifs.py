from typing import TYPE_CHECKING, Any, TypeVar

from grgg.models.ergm.abc import AbstractErgmNodeMotifs
from grgg.statistics.motifs import (
    QHeadMotif,
    QuadrangleMotif,
    QWedgeMotif,
    THeadMotif,
    TriangleMotif,
    TWedgeMotif,
)

from .qhead import UndirectedRandomGraphQHeadMotif
from .quadrangle import UndirectedRandomGraphQuadrangleMotif
from .qwedge import UndirectedRandomGraphQWedgeMotif
from .thead import UndirectedRandomGraphTHeadMotif
from .triangle import UndirectedRandomGraphTriangleMotif
from .twedge import UndirectedRandomGraphTWedgeMotif

if TYPE_CHECKING:
    from grgg.models.ergm.random_graphs.undirected.model import UndirectedRandomGraph
    from grgg.models.ergm.random_graphs.undirected.views import (
        UndirectedRandomGraphNodePairView,
        UndirectedRandomGraphNodeView,
    )

    T = TypeVar("T", bound="UndirectedRandomGraph")
    V = TypeVar("V", bound="UndirectedRandomGraphNodeView")
    E = TypeVar("E", bound="UndirectedRandomGraphNodePairView")


class UndirectedRandomGraphNodeMotifs[V](AbstractErgmNodeMotifs[V]):
    """Class for undirected node motifs."""

    view: "UndirectedRandomGraphNodeView"


# Register motif statistics ----------------------------------------------------------


@TriangleMotif.from_module.register
@classmethod
def _(
    cls,  # noqa
    module: UndirectedRandomGraphNodeMotifs,  # noqa
    *args: Any,
    **kwargs: Any,
) -> UndirectedRandomGraphTriangleMotif:
    return UndirectedRandomGraphTriangleMotif(module, *args, **kwargs)


@TWedgeMotif.from_module.register
@classmethod
def _(
    cls,  # noqa
    module: UndirectedRandomGraphNodeMotifs,  # noqa
    *args: Any,
    **kwargs: Any,
) -> UndirectedRandomGraphTWedgeMotif:
    return UndirectedRandomGraphTWedgeMotif(module, *args, **kwargs)


@THeadMotif.from_module.register
@classmethod
def _(
    cls,  # noqa
    module: UndirectedRandomGraphNodeMotifs,  # noqa
    *args: Any,
    **kwargs: Any,
) -> UndirectedRandomGraphTHeadMotif:
    return UndirectedRandomGraphTHeadMotif(module, *args, **kwargs)


@QuadrangleMotif.from_module.register
@classmethod
def _(
    cls,  # noqa
    module: UndirectedRandomGraphNodeMotifs,  # noqa
    *args: Any,
    **kwargs: Any,
) -> UndirectedRandomGraphQuadrangleMotif:
    return UndirectedRandomGraphQuadrangleMotif(module, *args, **kwargs)


@QWedgeMotif.from_module.register
@classmethod
def _(
    cls,  # noqa
    module: UndirectedRandomGraphNodeMotifs,  # noqa
    *args: Any,
    **kwargs: Any,
) -> UndirectedRandomGraphQWedgeMotif:
    return UndirectedRandomGraphQWedgeMotif(module, *args, **kwargs)


@QHeadMotif.from_module.register
@classmethod
def _(
    cls,  # noqa
    module: UndirectedRandomGraphNodeMotifs,  # noqa
    *args: Any,
    **kwargs: Any,
) -> UndirectedRandomGraphQHeadMotif:
    return UndirectedRandomGraphQHeadMotif(module, *args, **kwargs)
