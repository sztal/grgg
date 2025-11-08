from abc import abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, TypeVar

import equinox as eqx
import numpy as np
from scipy.sparse import sparray

from grgg.models.abc import AbstractModelSampler

if TYPE_CHECKING:
    from .model import AbstractErgm
    from .views import AbstractErgmNodeView

    T = TypeVar("T", bound="AbstractErgm")
    NV = TypeVar("NV", bound="AbstractErgmNodeView")

    try:
        import igraph as ig
    except ImportError:  # pragma: no cover
        pass
    try:
        from pathcensus import PathCensus
    except ImportError:  # pragma: no cover
        pass
    try:
        import networkx as nx
    except ImportError:  # pragma: no cover
        pass

__all__ = ("AbstractErgmSampler", "ErgmSample")


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

    @cached_property
    def struct(self) -> "PathCensus":
        """Return the path census for calculating structural coefficients.

        See :mod:`pathcensus` for details.
        """
        try:
            from pathcensus import PathCensus
        except ImportError as exc:  # pragma: no cover
            errmsg = (
                "Path census requires 'pathcensus' package. "
                "Install it, e.g. with `pip install pathcensus`."
            )
            raise ImportError(errmsg) from exc
        return PathCensus(self.A)

    @cached_property
    def igraph(self) -> "ig.Graph":
        """Return the :mod:`igraph` representation of the sampled graph."""
        try:
            import igraph as ig
        except ImportError as exc:  # pragma: no cover
            errmsg = (
                "igraph representation requires 'python-igraph' package. "
                "Install it, e.g. with `pip install python-igraph`."
            )
            raise ImportError(errmsg) from exc
        # Make igraph graph from sparse adjacency matrix
        edges = np.column_stack(self.A.nonzero())
        G = ig.Graph(edges, directed=False, n=self.A.shape[0])
        return G.simplify()

    @property
    def ig(self) -> "ig.Graph":
        """Alias for igraph property."""
        return self.igraph

    @property
    def G(self):
        """Alias for igraph property."""
        return self.igraph

    @property
    def networkx(self) -> "nx.Graph":
        """Return the :mod:`networkx` representation of the sampled graph."""
        try:
            import networkx as nx
        except ImportError as exc:  # pragma: no cover
            errmsg = (
                "networkx representation requires 'networkx' package. "
                "Install it, e.g. with `pip install networkx`."
            )
            raise ImportError(errmsg) from exc
        return nx.from_scipy_sparse_array(self.A)

    @property
    def nx(self) -> "nx.Graph":
        """Alias for networkx property."""
        return self.networkx


class AbstractErgmSampler[T](AbstractModelSampler[T]):
    """Abstract base class for samplers of static graph models."""

    nodes: eqx.AbstractVar["NV"]

    @property
    def model(self) -> T:
        """Parent model."""
        return self.nodes.model

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the view."""
        return self.nodes.n_nodes

    @abstractmethod
    def sample(self, *args: Any, **kwargs: Any) -> ErgmSample:
        """Sample a graph from the model."""
