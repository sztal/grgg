from abc import abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, TypeVar

import equinox as eqx
import numpy as np
from scipy.sparse import sparray

from grgg.models.abc import AbstractModelSampler

if TYPE_CHECKING:
    from .models import AbstractErgm
    from .views import AbstractErgmNodeView

__all__ = ("AbstractErgmSampler", "ErgmSample")


T = TypeVar("T", bound="AbstractErgm")
V = TypeVar("V", bound="AbstractErgmNodeView")
S = TypeVar("S", bound="ErgmSample")


@dataclass(frozen=True)
class ErgmSample:
    """Sample from a models.

    Attributes
    ----------
    A
        Sparse adjacency matrix of the sample graph.
    G
        :mod:`igraph` representation of the sample graph.
        Requires :mod:`igraph` to be installed.
    struct
        Path census of the sample graph.
        Requires :mod:`pathcensus` to be installed.
    """

    A: sparray

    try:
        from pathcensus import PathCensus

        @cached_property
        def struct(self) -> PathCensus:  # type: ignore
            """Return the path census for calculating structural coefficients.

            See :mod:`pathcensus` for details.
            """
            from pathcensus import PathCensus  # noqa

            return PathCensus(self.A)
    except ImportError:  # pragma: no cover

        @cached_property
        def struct(self) -> None:
            errmsg = (
                "Path census requires `pathcensus` package. "
                "Install it with `pip install pathcensus`."
            )
            raise ImportError(errmsg)

    try:
        import igraph as ig

        @cached_property
        def igraph(self) -> ig.Graph:  # type: ignore
            """Return the :mod:`igraph` representation of the sampled graph."""
            import igraph as ig  # noqa

            # Make igraph graph from sparse adjacency matrix
            edges = np.column_stack(self.A.nonzero())
            G = ig.Graph(edges, directed=False, n=self.A.shape[0])
            return G.simplify()
    except ImportError:  # pragma: no cover

        @cached_property
        def igraph(self) -> None:
            errmsg = (
                "igraph representation requires `python-igraph` package. "
                "Install it with `pip install python-igraph`."
            )
            raise ImportError(errmsg)

    @property
    def G(self):
        """Alias for igraph property."""
        return self.igraph


class AbstractErgmSampler[T, V, S](AbstractModelSampler[T]):
    """Abstract base class for samplers of static graph models."""

    nodes: eqx.AbstractVar[V]

    @property
    def model(self) -> T:
        """Parent model."""
        return self.nodes.model

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the view."""
        return self.nodes.n_nodes

    @abstractmethod
    def sample(self, *args: Any, **kwargs: Any) -> S:
        """Sample a graph from the model."""
