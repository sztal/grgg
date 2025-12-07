from collections.abc import Callable
from typing import Annotated, Any, ClassVar, Literal

import equinox as eqx
import jax.numpy as jnp

from grgg._typing import IsHeterogeneous, IsHomogeneous, Real, Reals
from grgg.models.base.model import Parameters
from grgg.models.base.random_graphs import AbstractRandomGraph, Mu
from grgg.models.base.traits import Undirected, Unweighted

from .functions import RandomGraphFunctions
from .views import RandomGraphNodePairView, RandomGraphNodeView

__all__ = ("RandomGraph",)


class RandomGraph(AbstractRandomGraph, Undirected, Unweighted):
    """Undirected random graph model.

    This class implements an exponential random graph model (ERGM) for undirected
    unweighted graphs. It is equivalent to the (n, p)-Erdős-Rényi model when ``mu``
    is homogeneous (scalar), or to the soft configuration model when ``mu`` is
    heterogeneous (node-specific vector).

    The model is defined by the edge probability function:

    .. math::

        p_{ij} = \\frac{1}{1 + \\exp(-\\mu_i - \\mu_j)}

    where :math:`\\mu_i` is the chemical potential of node :math:`i`. When ``mu``
    is homogeneous (:math:`\\mu_i = \\mu` for all :math:`i`), this reduces to
    :math:`p_{ij} = p` for all pairs, yielding the classical Erdős-Rényi model.

    Attributes
    ----------
    n_nodes
        Number of nodes in the graph.
    mu
        Chemical potential parameter(s) controlling node degrees. Can be:
        - Scalar (homogeneous): Same edge probability for all pairs
        - Vector of length ``n_nodes`` (heterogeneous): Node-specific parameters

    Notes
    -----
    The model supports both exact analytical computations and Monte Carlo estimation
    of expected network statistics. Use the ``nodes`` and ``pairs`` views to access
    various statistics like degree, clustering, and motif counts.

    Examples
    --------
    **Homogeneous case (Erdős-Rényi model)**

    Create a homogeneous random graph model with 100 nodes and uniform edge
    probability:

    >>> import jax.numpy as jnp
    >>> from grgg import RandomGraph, RandomGenerator
    >>> rng = RandomGenerator(42)
    >>> model = RandomGraph(100, mu=-2.0)

    The ``mu`` parameter is homogeneous (scalar):

    >>> model.parameters.mu.is_homogeneous
    True
    >>> model.parameters.mu.data
    Array(-2.0, ...)

    Sample a graph instance:

    >>> sample = model.sample(rng=rng)
    >>> sample.A.shape  # adjacency matrix
    (100, 100)
    >>> sample.G.vcount()  # igraph.Graph object
    100

    Check expected vs. observed average degree:

    >>> expected_degree = model.nodes.degree().mean()
    >>> observed_degree = jnp.asarray(sample.G.degree()).mean()
    >>> jnp.isclose(expected_degree, observed_degree, rtol=0.2).item()
    True

    **Heterogeneous case (soft configuration model)**

    Create a heterogeneous random graph model with node-specific parameters:

    >>> n = 100
    >>> mu_values = rng.normal(n) - 2.5
    >>> model_het = RandomGraph(n, mu=mu_values)

    The ``mu`` parameter is heterogeneous (vector):

    >>> model_het.parameters.mu.is_heterogeneous
    True
    >>> model_het.parameters.mu.data.shape
    (100,)

    Sample and verify degree distribution follows the expected pattern:

    >>> sample_het = model_het.sample(rng=rng)
    >>> expected_degrees = model_het.nodes.degree()
    >>> observed_degrees = jnp.asarray(sample_het.G.degree())
    >>> correlation = jnp.corrcoef(expected_degrees, observed_degrees)[0, 1]
    >>> (correlation > 0.8).item()  # high correlation expected
    True

    **Model fitting**

    The model can be fit to observed graph data to infer parameters:

    >>> import igraph as ig
    >>> G = ig.Graph.Erdos_Renyi(n=100, p=0.1)
    >>> model = RandomGraph(100)
    >>> fit = model.fit(G, homogeneous=True)
    >>> fit.model.parameters.mu.is_homogeneous  # infers homogeneous structure
    True
    >>> jnp.isclose(fit.model.edge_density(), G.density()).item()
    True
    """

    class Parameters(Parameters):
        mu: Mu = eqx.field(default_factory=lambda: Mu(), converter=Mu)

    n_nodes: int = eqx.field(static=True)
    parameters: Parameters

    functions: ClassVar[type[RandomGraphFunctions]] = RandomGraphFunctions

    nodes_cls: ClassVar[type[RandomGraphNodeView]] = RandomGraphNodeView
    pairs_cls: ClassVar[type[RandomGraphNodePairView]] = RandomGraphNodePairView

    def __init__(self, n_nodes: int, **kwargs: Any) -> None:
        self.n_nodes = n_nodes
        super().__init__(**kwargs)

    # Model fitting interface --------------------------------------------------------

    @AbstractRandomGraph._init_param.dispatch
    def _(
        self: "RandomGraph",
        param: Mu,
        target: Reals,
        homogeneous: Literal[True],  # noqa
    ) -> Mu:
        return self._init_erdos_renyi(param, target)

    @AbstractRandomGraph._init_param.dispatch
    def _(
        self: "RandomGraph",
        param: Mu,
        target: Reals,
        homogeneous: Literal[False],  # noqa
    ) -> Mu:
        return self._init_chung_lu(param, target)

    @eqx.filter_jit
    def _init_chung_lu(
        self,
        param: Annotated[Mu, IsHeterogeneous],
        degree: Reals,
    ) -> Mu:
        """Chung-Lu initialization."""
        return param.replace(data=jnp.log(degree / jnp.sqrt(jnp.sum(degree))))

    @eqx.filter_jit
    def _init_erdos_renyi(
        self,
        param: Annotated[Mu, IsHomogeneous],
        ecount: Real,
    ) -> Mu:
        """Erdős–Rényi initialization."""
        n = self.n_nodes
        p = ecount / (n * (n - 1))
        if self.is_undirected:
            p *= 2
        theta = jnp.log((1 - p) / p)
        return param.replace(data=-theta)


# Mapping between parameters and observables -----------------------------------------


@Mu._get_statistic.dispatch
def _(
    self,  # noqa
    model: RandomGraph,
    homogeneous: Literal[True],  # noqa
    method: Literal["lagrangian", "least_squares"],  # noqa
) -> tuple[Literal["edge_count"], Callable[..., Real]]:
    return "edge_count", model.edge_count


@Mu._get_statistic.dispatch
def _(
    self,  # noqa
    model: RandomGraph,
    homogeneous: Literal[False],  # noqa
    method: Literal["lagrangian", "least_squares"],  # noqa
) -> tuple[Literal["degree"], Callable[..., Reals]]:
    return "degree", model.nodes.degree
