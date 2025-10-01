from typing import TYPE_CHECKING, Any, TypeVar

from grgg.models.ergm.abc import AbstractErgmNodeMotifs
from grgg.statistics.motifs import (
    QHeadMotifStatistic,
    QuadrangleMotifStatistic,
    QWedgeMotifStatistic,
    THeadMotifStatistic,
    TriangleMotifStatistic,
    TWedgeMotifStatistic,
)

from .statistics import (
    UndirectedRandomGraphQHeadMotifStatistic,
    UndirectedRandomGraphQuadrangleMotifStatistic,
    UndirectedRandomGraphQWedgeMotifStatistic,
    UndirectedRandomGraphTHeadMotifStatistic,
    UndirectedRandomGraphTriangleMotifStatistic,
    UndirectedRandomGraphTWedgeMotifStatistic,
)

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


@TriangleMotifStatistic.from_module.register
@classmethod
def _(
    cls,  # noqa
    module: UndirectedRandomGraphNodeMotifs,  # noqa
    *args: Any,
    **kwargs: Any,
) -> UndirectedRandomGraphTriangleMotifStatistic:
    return UndirectedRandomGraphTriangleMotifStatistic(module, *args, **kwargs)


@TWedgeMotifStatistic.from_module.register
@classmethod
def _(
    cls,  # noqa
    module: UndirectedRandomGraphNodeMotifs,  # noqa
    *args: Any,
    **kwargs: Any,
) -> UndirectedRandomGraphTWedgeMotifStatistic:
    return UndirectedRandomGraphTWedgeMotifStatistic(module, *args, **kwargs)


@THeadMotifStatistic.from_module.register
@classmethod
def _(
    cls,  # noqa
    module: UndirectedRandomGraphNodeMotifs,  # noqa
    *args: Any,
    **kwargs: Any,
) -> UndirectedRandomGraphTHeadMotifStatistic:
    return UndirectedRandomGraphTHeadMotifStatistic(module, *args, **kwargs)


@QuadrangleMotifStatistic.from_module.register
@classmethod
def _(
    cls,  # noqa
    module: UndirectedRandomGraphNodeMotifs,  # noqa
    *args: Any,
    **kwargs: Any,
) -> UndirectedRandomGraphQuadrangleMotifStatistic:
    return UndirectedRandomGraphQuadrangleMotifStatistic(module, *args, **kwargs)


@QWedgeMotifStatistic.from_module.register
@classmethod
def _(
    cls,  # noqa
    module: UndirectedRandomGraphNodeMotifs,  # noqa
    *args: Any,
    **kwargs: Any,
) -> UndirectedRandomGraphQWedgeMotifStatistic:
    return UndirectedRandomGraphQWedgeMotifStatistic(module, *args, **kwargs)


@QHeadMotifStatistic.from_module.register
@classmethod
def _(
    cls,  # noqa
    module: UndirectedRandomGraphNodeMotifs,  # noqa
    *args: Any,
    **kwargs: Any,
) -> UndirectedRandomGraphQHeadMotifStatistic:
    return UndirectedRandomGraphQHeadMotifStatistic(module, *args, **kwargs)
