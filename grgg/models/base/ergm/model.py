from dataclasses import dataclass
from functools import cached_property, wraps
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import equinox as eqx
import numpy as np
from scipy.sparse import sparray

from grgg._typing import Real, Reals
from grgg.statistics import EdgeCount

from ..model import AbstractModel
from ..traits import EdgeDirection, EdgeWeighting
from .fitting import ExpectedStatistics, LagrangianFit, SufficientStatistics
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

    # Statistics ---------------------------------------------------------------------

    @property
    @wraps(EdgeCount.__init__)
    def edge_count(self) -> EdgeCount:
        """Edge count statistic of the model.

        Examples
        --------
        >>> import igraph as ig
        >>> import jax.numpy as jnp
        >>> from grgg import RandomGraph
        >>> n = 100
        >>> model = RandomGraph(n)
        >>> degree = model.nodes.degree()
        >>> jnp.isclose(model.edge_count(), degree * n / 2).item()
        True
        >>> model = RandomGraph(n, mu=jnp.zeros(n))
        >>> degree = model.nodes.degree()
        >>> jnp.isclose(model.edge_count(), degree.sum() / 2).item()
        True
        >>> G = ig.Graph.Full(n)
        >>> (model.edge_count.observed(G) == G.ecount()).item()
        True
        """
        return EdgeCount(self)

    def edge_density(self, *args: Any, **kwargs: Any) -> float:
        """Expected edge density of the model."""
        n = self.n_nodes
        ecount = self.edge_count(*args, **kwargs).mean() / (self.n_nodes - 1)
        density = ecount / (n * (n - 1))
        if self.is_undirected:
            density *= 2
        return density

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

    @AbstractModel.fit.dispatch
    def _(self: "AbstractErgm", target: SufficientStatistics) -> LagrangianFit:
        return LagrangianFit(self, target)

    @AbstractModel.get_target_cls.dispatch
    def _(
        self: "AbstractErgm",
        method: Literal["lagrangian"],  # noqa
    ) -> type[SufficientStatistics]:
        return SufficientStatistics

    @AbstractModel.get_target_cls.dispatch
    def _(
        self: "AbstractErgm",
        method: Literal["least_squares"],  # noqa
    ) -> type[ExpectedStatistics]:
        return ExpectedStatistics


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
