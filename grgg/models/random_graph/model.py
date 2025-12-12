from collections.abc import Callable
from typing import Any, ClassVar, Literal

import equinox as eqx
import jax.numpy as jnp

from grgg._typing import Real, Reals
from grgg.models.base.model import AbstractObservables, AbstractParameters
from grgg.models.base.random_graphs import AbstractRandomGraph, Mu
from grgg.models.base.traits import Undirected, Unweighted
from grgg.utils.random import RandomGenerator

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
    Array(-2., ...)

    Sample a graph instance:

    >>> sample = model.sample(rng=rng)
    >>> sample.A.shape  # adjacency matrix
    (100, 100)
    >>> sample.G.vcount()  # igraph.Graph object
    100
    >>> sample.nx.number_of_nodes()  # networkx.Graph object
    100
    >>> sample.struct.vcount  # pathcensus.PathCensus object
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
    >>> fit = model.fit(G, "homogeneous")
    >>> fit.model.parameters.mu.is_homogeneous  # infers homogeneous structure
    True
    >>> jnp.isclose(fit.model.edge_density(), G.density(), rtol=1e-3).item()
    True

    Fit a heterogeneous model to match observed degree sequence:

    >>> fit = model.fit(G, "heterogeneous")
    >>> fit.model.parameters.mu.is_heterogeneous  # infers heterogeneous structure
    True
    >>> expected_degrees = fit.model.nodes.degree()
    >>> observed_degrees = jnp.asarray(G.degree())
    >>> (jnp.corrcoef(expected_degrees, observed_degrees)[0, 1] > 0.95).item()
    True

    Heterogeneous model can also be initialized for fitting with random values in
    [-1, 1] (with possible control over the random generator) or just zeros.
    >>> from grgg import RandomGenerator
    >>> rng = RandomGenerator(123)
    >>> fit_random = model.fit(G, mu="random", rng=rng)
    >>> mu = fit_random.model.parameters.mu.data
    >>> (jnp.all(mu >= -1.0) and jnp.all(mu <= 1.0)).item()
    True
    >>> fit_zeros = model.fit(G, mu="zeros")
    >>> mu = fit_zeros.model.parameters.mu.data
    >>> (jnp.all(mu == 0.0)).item()
    True
    """

    class Parameters(AbstractParameters):
        mu: Mu = eqx.field(default_factory=lambda: Mu(), converter=Mu)

    class Observables(AbstractObservables):
        degree: Reals

        @property
        def names(self) -> list[str]:
            return [*super().names, "edge_count"]

        @property
        def edge_count(self) -> Real:
            """Total number of edges."""
            return jnp.sum(self.degree) / 2

    n_nodes: int = eqx.field(static=True)
    parameters: Parameters

    functions: ClassVar[type[RandomGraphFunctions]] = RandomGraphFunctions

    nodes_cls: ClassVar[type[RandomGraphNodeView]] = RandomGraphNodeView
    pairs_cls: ClassVar[type[RandomGraphNodePairView]] = RandomGraphNodePairView

    def __init__(self, n_nodes: int, **kwargs: Any) -> None:
        self.n_nodes = n_nodes
        super().__init__(**kwargs)


# Parameters' initialization interface -----------------------------------------------


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


@Mu.initialize.dispatch
def _(
    self: Mu,  # noqa
    model: RandomGraph,  # noqa
    target: RandomGraph.Observables,
    method: Literal["homogeneous", "erdos_renyi", "er", "erdos"],  # noqa
) -> Mu:
    n = model.n_nodes
    ecount = target.edge_count
    p = 2 * jnp.asarray(ecount / n / (n - 1))
    theta = jnp.log((1 - p) / p)
    return self.replace(data=-theta)


@Mu.initialize.dispatch
def _(
    self: Mu,  # noqa
    model: RandomGraph,  # noqa
    target: RandomGraph.Observables,
    method: Literal["heterogeneous", "chung-lu", "configuration-model", "cm"],  # noqa
) -> Mu:
    degree = target.degree
    return self.replace(data=jnp.log(degree / jnp.sqrt(jnp.sum(degree))))


@Mu.initialize.dispatch
def _(
    self: Mu,  # noqa
    model: RandomGraph,  # noqa
    target: RandomGraph.Observables,  # noqa
    method: Literal["random"],  # noqa
    rng: RandomGenerator | None = None,
) -> Mu:
    rng = RandomGenerator(rng)
    theta = rng.uniform(model.n_nodes, minval=-1.0, maxval=1.0)
    return self.replace(data=-theta)


@Mu.initialize.dispatch
def _(
    self: Mu,  # noqa
    model: RandomGraph,  # noqa
    target: RandomGraph.Observables,  # noqa
    method: Literal["zeros"],  # noqa
) -> Mu:
    return self.replace(data=jnp.zeros(model.n_nodes))
