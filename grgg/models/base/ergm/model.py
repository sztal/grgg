from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, TypeVar

import equinox as eqx
import numpy as np
from scipy.sparse import sparray

from grgg._typing import Real, Reals
from grgg.utils.dispatch import dispatch

from ..model import AbstractModel
from ..traits import EdgeDirection, EdgeWeighting
from .fitting import AbstractSufficientStatistics, LagrangianFit
from .functions import AbstractErgmFunctions
from .views import AbstractErgmNodePairView, AbstractErgmNodeView

if TYPE_CHECKING:
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

__all__ = ("AbstractErgm", "ErgmSample")


T = TypeVar("T", bound="AbstractErgm")
NV = TypeVar("NV", bound="AbstractErgmNodeView")


class AbstractErgm(AbstractModel, EdgeDirection, EdgeWeighting):
    """Abstract base class for ERGMs."""

    n_nodes: eqx.AbstractVar[int]

    functions: eqx.AbstractClassVar[type[AbstractErgmFunctions]]

    nodes_cls: eqx.AbstractClassVar[type[AbstractErgmNodeView]]
    pairs_cls: eqx.AbstractClassVar[type[AbstractErgmNodePairView]]

    def __check_init__(self) -> None:
        if self.n_nodes <= 0:
            errmsg = f"'n_nodes' must be positive, got {self.n_nodes}."
            raise ValueError(errmsg)

    @property
    def n_units(self) -> int:
        """Number of units in the model."""
        return self.n_nodes

    @property
    def is_undirected(self) -> bool:
        """Whether the model is undirected."""
        return not self.is_directed

    @property
    def is_unweighted(self) -> bool:
        """Whether the model is unweighted."""
        return not self.is_weighted

    @property
    def nodes(self) -> AbstractErgmNodeView:
        """Node view of the model."""
        return self.nodes_cls(self)

    @property
    def pairs(self) -> AbstractErgmNodePairView:
        """Node pair view of the model."""
        return self.pairs_cls(self)

    # Model functions ----------------------------------------------------------------

    def free_energy(self, *args: Any, **kwargs: Any) -> Reals:
        """Compute the free energy of the model."""
        return self.functions.free_energy(self, *args, **kwargs)

    def partition_function(self, *args: Any, **kwargs: Any) -> Reals:
        """Compute the partition function of the model."""
        return self.functions.partition_function(self, *args, **kwargs)

    def hamiltonian(self, obj: Any, **kwargs: Any) -> Real:
        """Compute the Hamiltonian of the model."""
        return self.functions.hamiltonian(self, obj, **kwargs)

    def lagrangian(self, obj: Any, **kwargs: Any) -> Real:
        """Compute the Lagrangian of the model."""
        return self.functions.lagrangian(self, obj, **kwargs)

    # Model fitting interface --------------------------------------------------------

    @dispatch
    def fit(self, target: AbstractSufficientStatistics) -> LagrangianFit:
        return LagrangianFit(self, target)


# ERGM Sample ------------------------------------------------------------------------


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
