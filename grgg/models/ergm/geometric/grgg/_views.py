from collections.abc import Mapping, Sequence
from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax.numpy as jnp

from grgg.models.abc import (
    AbstractStaticGraphModelNodePairView,
    AbstractStaticGraphModelNodeView,
)

# from grgg.models.geometric.abc import AbstractGeometricGraph
# from grgg.utils.lazy import LazyOuter
from grgg.utils.misc import split_kwargs_by_signature, squareform

from ._sampling import GeometricSample, GeometricSampler
from .integrals import DegreeIntegral, EdgeProbabilityIntegral

LazyOuter = object
AbstractGeometricGraph = object

if TYPE_CHECKING:
    from .model import GRGG


class NodeView(AbstractStaticGraphModelNodeView):
    """Node view.

    Helper class for indexing model parameters and computing node-specific
    quantities for specific node selections.

    Attributes
    ----------
    module
        Parent model module.
    """

    @property
    def _pair_view_type(self) -> type["NodePairView"]:
        return NodePairView

    @property
    def beta(self) -> jnp.ndarray:
        """Beta parameter outer product."""
        return self._get_param(self.model.parameters, "beta")

    @property
    def mu(self) -> jnp.ndarray:
        """Mu parameter outer product."""
        return self._get_param(self.model.parameters, "mu")

    def sample_points(self, **kwargs: Any) -> jnp.ndarray:
        """Sample points from the selected group of nodes.

        `**kwargs`* are passed to :meth:`~grgg.manifolds.Manifold.sample_points`.
        """
        return self.model.manifold.sample_points(self.n_nodes, **kwargs)

    def sample_pmatrix(
        self,
        points: jnp.ndarray | None = None,
        *,
        condensed: bool = False,
        **kwargs: Any,
    ) -> jnp.ndarray:
        """Sample probability matrix from the selected group of nodes.

        Parameters
        ----------
        points
            Points to use. If `None`, points are sampled using
            :meth:`~grgg.model._views.NodeView.sample_points`.
        condensed
            Whether to return the condensed form of the probability matrix.
        **kwargs
            Additional arguments passed to
            :meth:`~grgg.model._views.NodeView.sample_points`.

        Examples
        --------
        >>> from grgg import GRGG, Similarity, RandomGenerator
        >>> rng = RandomGenerator(0)
        >>> model = GRGG(5, 2, Similarity(2, 1))
        >>> P = model.nodes.sample_pmatrix(rng=rng)  # Full probability matrix
        >>> P.shape
        (5, 5)
        >>> P_sub = model.nodes[[0, 2, 4]].sample_pmatrix(rng=rng)  # nodes 0, 2, and 4
        >>> P_sub.shape
        (3, 3)
        """
        self = self.materialize(copy=False).nodes if self.is_active else self
        if points is None:
            points = self.sample_points(**kwargs)
        g = self.model.manifold.distances(points, condensed=True)
        i, j = jnp.triu_indices(len(points), k=1)
        p = self.pairs[i, j].probs(g)
        return p if condensed else squareform(p)

    def sample(self, **kwargs: Any) -> GeometricSample:
        """Generate a model sample for the selected group of nodes.

        `**kwargs`* are passed to :meth:`~grgg.model._sampling.Sampler.sample`."""
        return GeometricSampler(self).sample(**kwargs)

    def materialize(self, *, copy: bool = False) -> "GRGG":
        """Materialize a new GRGG model with only the selected nodes.

        Parameters
        ----------
        copy
            Whether to return a deep copy of the model.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from grgg import GRGG, Similarity, Complementarity
        >>> model = GRGG(100, 2, Similarity(2, jnp.zeros(100)), Complementarity(1, 0))
        >>> submodel = model.nodes[:10].materialize()
        >>> submodel.n_nodes
        10
        >>> submodel.manifold.volume
        100.0
        >>> submodel.layers[0].beta.shape
        (10,)
        """
        if not isinstance(self.model, "AbstractGeometricGraph"):
            errmsg = "only views of the full GRGG model can be materialized"
            raise TypeError(errmsg)
        if self._index is None:
            return self.model
        index = self.index
        if isinstance(index, tuple):
            if len(index) != 1:
                errmsg = "only single-axis indexing is supported for node views"
                raise IndexError(errmsg)
            index = index[0]
        if isinstance(index, jnp.ndarray) and index.ndim > 1:
            errmsg = "only 1D array indexing is supported when materializing node views"
            raise IndexError(errmsg)
        model = self.model.copy(deep=True) if copy else self.model
        layers = [
            layer.copy()
            .detach()
            .replace(
                beta=beta.copy() if copy else beta,
                mu=mu.copy() if copy else mu,
            )
            for layer, beta, mu in zip(model.layers, self.beta, self.mu, strict=False)
        ]
        return model.replace(
            n_nodes=self.shape[0],
            manifold=model.manifold,
            layers=layers,
        )

    def degree(
        self,
        *args: Any,
        full_shape: bool = False,
        dequantize: bool = True,
        **kwargs: Any,
    ) -> jnp.ndarray:
        """Compute expected degrees for the selected nodes.

        Parameters
        ----------
        *args, **kwargs
            Arguments passed to :class:`~grgg.model.integrals.DegreeIntegral`
            and :meth:`~grgg.model.integrals.DegreeIntegral.integrate`.
        full_shape
            Whether to return the output in the full shape for homogeneous models.
        dequantized
            Whether to dequantize the output when computing degrees based
            on the quantized model.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from grgg import GRGG, Similarity, RandomGenerator
        >>> model = GRGG(100, 2, Similarity(2, 1))
        >>> D = model.nodes.degree()
        >>> D.shape
        ()
        >>> D = model.nodes.degree(full_shape=True)  # full shape for homogeneous models
        >>> D.shape
        (100,)
        >>> D = model.nodes[10:20].degree(full_shape=True)  # selected nodes
        >>> D.shape
        (10,)

        Degree calculations are supported for quantized models too.
        By default, the output is dequantized.
        >>> rng = RandomGenerator(17)
        >>> n = 100
        >>> model = GRGG(n, 2, Similarity(rng.normal(n), rng.normal(n)**2))
        >>> qmodel = model.quantize(n_codes=32, random_state=17)
        >>> D = model.nodes.degree()
        >>> Q = qmodel.nodes.degree()
        >>> Q.shape
        (100,)
        >>> jnp.corrcoef(D, Q)[0, 1].item()  # correlation with non-quantized degrees
        0.9181299
        >>> jnp.abs((D - Q) / D).mean().item()  # mean relative error
        0.1894010

        If `dequantize=False`, the output is not dequantized.
        >>> Q = qmodel.nodes.degree(dequantize=False)
        >>> Q.shape
        (32,)
        """
        init_kwargs, kwargs = split_kwargs_by_signature(DegreeIntegral, **kwargs)
        integrator = DegreeIntegral.from_nodes(self, **init_kwargs)
        integral = integrator.integrate(*args, **kwargs)[0]
        if self.model.is_homogeneous:
            return jnp.full((self.n_nodes,), integral) if full_shape else integral
        if self.model.is_quantized and dequantize:
            integral = self.model.dequantize_arrays(integral)[0]
        return integral

    @singledispatchmethod
    def _get_param(self, params: Mapping, name: str) -> jnp.ndarray:
        """Get parameter."""
        param = params[name]
        if self._index is None or jnp.isscalar(param):
            return param
        return param[self.index]

    @_get_param.register
    def _(self, params: Sequence, name: str) -> list[jnp.ndarray]:
        return [self._get_param(p, name) for p in params]


class NodePairView(AbstractStaticGraphModelNodePairView):
    """Node pairs view.

    Helper class for indexing model parameters and computing pairwise
    connection probabilities and other quantities for specific node pair
    selections.

    Attributes
    ----------
    module
        Parent model module.

    Examples
    --------
    Indexing in the homogeneous case always returns scalars.
    >>> import jax.numpy as jnp
    >>> from grgg import GRGG, Similarity, Complementarity
    >>> model = GRGG(100, 2, Similarity(1, 2), Complementarity(0, 1))
    >>> model.pairs[0, 1].beta
    [Array(2., ...), Array(1., ...]
    >>> model.pairs[[0, 1], [1, 0]].mu
    [Array(1., ...), Array(0., ...)]

    A specific layer can be indexed too.
    >>> model.layers[0].pairs[0, 1].beta
    Array(2., ...)

    In the heterogeneous case, indexing may return larger arrays.
    >>> model = GRGG(3, 2, Similarity([4,5,6], [1,2,3]))
    >>> model.pairs[0, 1].beta
    [Array(3., ...)]
    >>> model.pairs[...].mu
    [Array([[ 8.,  9., 10.],
            [ 9., 10., 11.],
            [10., 11., 12.]], ...)]

    Arbitrary NumPy-style indexing is supported.
    Below we select pairs (0,1) and (1,2).
    >>> model.layers[0].pairs[[0, 1], [1, 2]].mu
    Array([ 9., 11.], ...)

    Selecting rectangular is also possible, either through basic slicing,
    or index cross products.
    >>> model.layers[0].pairs[:2, 1:3].mu
    Array([[ 9., 10.],
           [10., 11.]], ...)
    >>> i = jnp.array([0, 1])
    >>> j = jnp.array([0, 1, 2])
    >>> model.layers[0].pairs[jnp.ix_(i, j)].beta
    Array([[2., 3., 4.],
           [3., 4., 5.]], ...)
    """

    @property
    def beta(self) -> LazyOuter | list[LazyOuter]:
        """Beta parameter outer product."""
        return self._get_param(self.model.parameters, "beta")

    @property
    def mu(self) -> LazyOuter | list[LazyOuter]:
        """Mu parameter outer product."""
        return self._get_param(self.model.parameters, "mu")

    def probs(self, g: jnp.ndarray, adjust_quantized: bool = False) -> jnp.ndarray:
        """Compute pairwise connection probabilities.

        Parameters
        ----------
        g
            Pairwise distances.
        adjust_quantized
            boolean flag to indicate whether to adjust for self-loops when
            computing connection probabilities between bins in quantized models.

        Examples
        --------
        For homogeneous models, indexing always returns scalars.
        We use non-logged and non-modified energies to make it easier
        to verify the results.
        >>> import jax.numpy as jnp
        >>> from jax.scipy.special import expit
        >>> from grgg import GRGG, Similarity, Complementarity, options
        >>> options.model.log = False
        >>> options.model.modified = False
        >>> model = (
        ...     GRGG(100, 2) +
        ...     Similarity(2, 1) +
        ...     Complementarity(3, 2)
        ... )
        >>> d = model.manifold.dim
        >>> gmax = model.manifold.diameter
        >>> g = jnp.linspace(0, gmax, 10)
        >>> tol = {"rtol": 1e-4, "atol": 1e-4}

        Check similarity layer.
        >>> mu, beta = model.parameters[0].values()
        >>> sim_probs = model[0].pairs.probs(g)
        >>> expected = expit(-d*beta*(g - mu))
        >>> jnp.allclose(sim_probs, expected, **tol).item()
        True

        Check complementarity layer.
        >>> mu, beta = model.parameters[1].values()
        >>> comp_probs = model[1].pairs.probs(g)
        >>> expected = expit(-d*beta*(gmax - g - mu))
        >>> jnp.allclose(comp_probs, expected, **tol).item()
        True

        Check multilayer model.
        >>> probs = model.pairs.probs(g)
        >>> jnp.allclose(probs, 1 - (1-sim_probs)*(1-comp_probs), **tol).item()
        True

        For heterogeneous models arrays are returned.
        >>> model = (
        ...     GRGG(3, 3) +
        ...     Similarity([1,2,3], 1) +
        ...     Complementarity([4,5,6], [0,1,2])
        ... )
        >>> gmax = model.manifold.diameter
        >>> g = jnp.linspace(0, gmax, 10)
        >>> d = model.manifold.dim

        Check similarity layer.
        >>> params = model.parameters[0]
        >>> mu = params.outer.mu[...]
        >>> beta = params.outer.beta[...]
        >>> sim_probs = model[0].pairs.probs(g[:, None, None])
        >>> expected = expit(-d*beta*(g[:, None, None] - mu))
        >>> i, j = model.pairs.coords[...]  # zero out diagonals
        >>> expected = jnp.where(i == j, 0.0, expected)
        >>> sim_probs.shape == expected.shape
        True
        >>> jnp.allclose(sim_probs, expected, **tol).item()
        True

        Check complementarity layer.
        >>> params = model.parameters[1]
        >>> mu = params.outer.mu[...]
        >>> beta = params.outer.beta[...]
        >>> comp_probs = model[1].pairs.probs(g[:, None, None])
        >>> expected = expit(-d*beta*(gmax - g[:, None, None] - mu))
        >>> i, j = model.pairs.coords[...]  # zero out diagonals
        >>> expected = jnp.where(i == j, 0.0, expected)
        >>> comp_probs.shape == expected.shape
        True
        >>> jnp.allclose(comp_probs, expected, **tol).item()
        True

        Check multilayer model.
        >>> probs = model.pairs.probs(g[:, None, None])
        >>> expected = 1 - (1-sim_probs)*(1-comp_probs)
        >>> i, j = model.pairs.coords[...]  # zero out diagonals
        >>> expected = jnp.where(i == j, 0.0, expected)
        >>> probs.shape == expected.shape
        True
        >>> jnp.allclose(probs, expected, **tol).item()
        True

        Node pair indexing is supported too.
        Below we select pairs (0,1) and (2,1).
        >>> i, j = [0, 2], [1, 1]

        Check similarity layer.
        >>> params = model.parameters[0]
        >>> mu = params.outer.mu[i, j]
        >>> beta = params.outer.beta[i, j]
        >>> sim_probs = model[0].pairs[i, j].probs(g[:, None])
        >>> expected = expit(-d*beta*(g[:, None] - mu))
        >>> sim_probs.shape == expected.shape
        True
        >>> jnp.allclose(sim_probs, expected, **tol).item()
        True

        Check complementarity layer.
        >>> params = model.parameters[1]
        >>> mu = params.outer.mu[i, j]
        >>> beta = params.outer.beta[i, j]
        >>> comp_probs = model[1].pairs[i, j].probs(g[:, None])
        >>> expected = expit(-d*beta*(gmax - g[:, None] - mu))
        >>> comp_probs.shape == expected.shape
        True
        >>> jnp.allclose(comp_probs, expected, **tol).item()
        True

        Check multilayer model.
        >>> probs = model.pairs[i, j].probs(g[:, None])
        >>> expected = 1 - (1-sim_probs)*(1-comp_probs)
        >>> probs.shape == expected.shape
        True
        >>> jnp.allclose(probs, expected, **tol).item()
        True

        Note that when pairs are selected, the self-loop probabilities are still
        zeroed out correctly.
        >>> i, j = [1, 2], [1, 2]
        >>> jnp.all(model.pairs[i, j].probs(g[:, None]) == 0).item()
        True
        >>> jnp.all(model.pairs[2, 2].probs(g) == 0).item()
        True

        Reset options to original values.
        >>> options.reset()
        """
        return _pairs_probs(self, g, adjust_quantized=adjust_quantized)

    def expected_probs(
        self,
        *args: Any,
        full_shape: bool = False,
        dequantize: bool = True,
        **kwargs: Any,
    ) -> jnp.ndarray:
        """Connection probabilities averaged over all possible distances.

        Parameters
        ----------
        *args, **kwargs
            Arguments passed to :class:`~grgg.model.integrals.EdgeProbabilityIntegral`
            and :meth:`~grgg.model.integrals.EdgeProbabilityIntegral.integrate`.
        full_shape
            Whether to return the output in the full shape for homogeneous models.
        dequantized
            Whether to dequantize the output when computing probabilities based
            on the quantized model.

        Examples
        --------
        >>> from grgg import GRGG, Similarity, Complementarity, RandomGenerator
        >>> rng = RandomGenerator(42)
        >>> n = 100
        >>> model = (
        ...     GRGG(n, 2) +
        ...     Similarity(rng.normal(n), rng.normal(n)**2) +
        ...     Complementarity(rng.normal(n), rng.normal(n)**2)
        ... )
        >>> probs = model.pairs.expected_probs()
        >>> probs.shape
        (100, 100)

        Arbitrary indexing is supported.
        >>> model.pairs[1, None, :10].expected_probs().shape
        (1, 10)

        Check that it is consistent with expected node degrees.
        >>> degrees = model.nodes.degree()
        >>> expected_probs = model.pairs.expected_probs()
        >>> jnp.allclose(degrees, expected_probs.sum(axis=1)).item()
        True
        """
        return _pairs_expected_probs(
            self,
            *args,
            full_shape=full_shape,
            dequantize=dequantize,
            **kwargs,
        )

    def edge_distance_density(
        self, g: jnp.ndarray, *args: Any, **kwargs: Any
    ) -> jnp.ndarray:
        """Edge distance density function.

        This is the probability density function over distances between connected nodes.

        Parameters
        ----------
        g
            Pairwise distances.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from grgg import GRGG, Similarity, RandomGenerator
        >>> rng = RandomGenerator(42)
        >>> n = 100
        >>> model = GRGG(n, 2, Similarity(rng.normal(n), rng.normal(n)**2))
        >>> gmax = model.manifold.diameter
        >>> g = jnp.linspace(0, gmax, 10)
        >>> density = model.pairs.edge_distance_density(g[:, None, None])
        >>> density.shape
        (10, 100, 100)

        Indexing is supported too.
        >>> density = model.pairs[0, :10].edge_distance_density(g)
        >>> density.shape
        (10,)

        Check that the density integrates to 1.
        >>> g = jnp.linspace(0, gmax, 2000)[:, None, None]
        >>> density = model.pairs.edge_distance_density(g)
        >>> integral = jnp.trapezoid(density, g, axis=0)
        >>> jnp.all(jnp.all(integral.diagonal() == 0)).item()  # self-loops are 0's
        True
        >>> offdiag = integral[~jnp.eye(n, dtype=bool)]
        >>> jnp.allclose(offdiag, 1.0, rtol=1e-2).item()
        True

        Check that it is consistent with empirical average distance of edges.
        >>> def sample_edge_distances(model):
        ...     sample = model.nodes.sample(rng=rng)
        ...     D = model.manifold.distances(sample.X, condensed=False)
        ...     D = jnp.where(sample.A.toarray() == 0, jnp.nan, D)
        ...     return D
        >>>

        For heterogeneous models, the rate of convergence to theoretical expectation
        may be much slower. Here we check that the relative error is below 20% for
        200 samples, but one can check that the error decreases with more samples.
        >>> Eg = (g*density).sum(0) / density.sum(0)
        >>> Mg = jnp.stack([sample_edge_distances(model) for _ in range(200)])
        >>> Mg = jnp.nanmean(Mg, axis=0)
        >>> eg = Eg[jnp.tril_indices_from(Eg, k=-1)]
        >>> mg = Mg[jnp.tril_indices_from(Mg, k=-1)]
        >>> mask = ~jnp.isnan(mg)
        >>> relerr = jnp.linalg.norm(eg[mask] - mg[mask]) / jnp.linalg.norm(eg[mask])
        >>> relerr.item() < 0.2
        True
        """
        return _pairs_edge_distance_density(self, g, *args, **kwargs)

    @singledispatchmethod
    def _get_param(self, params: Mapping, name: str) -> LazyOuter:
        outer = params.outer[name]
        if self._index is None:
            return outer[...]
        return outer[self.index]

    @_get_param.register
    def _(self, params: Sequence, name: str) -> list[jnp.ndarray]:
        return [self._get_param(p, name) for p in params]

    def _get_ij(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        args = tuple(a for a in (self._index or ()) if a is not None)
        coords = self.reindex[args].coords[...]
        if len(coords) < 2:
            coords = self.index
        return coords


@eqx.filter_jit
def _pairs_probs(
    pairs: NodePairView,
    g: jnp.ndarray,
    *,
    adjust_quantized: bool = False,
) -> jnp.ndarray:
    """Compute pairwise connection probabilities."""
    params = pairs.model.parameters.aligned
    mu = jnp.stack([pairs._get_param(p, "mu") for p in params])
    beta = jnp.stack([pairs._get_param(p, "beta") for p in params])
    probs = pairs.model(g, mu, beta)
    if pairs.model.is_homogeneous or (
        pairs.model.is_quantized and not adjust_quantized
    ):
        return probs
    try:
        i, j = pairs._get_ij()
    except ValueError:
        # This must be a single integer index
        if adjust_quantized and pairs.model.is_quantized:
            w = pairs.model.parameters.weights[pairs.index]
            return probs.at[pairs.index].mul(1 / w * (w - 1))
        return probs.at[pairs.index].set(0.0)
    if adjust_quantized and pairs.model.is_quantized:
        weights = pairs.model.parameters.weights
        wi = weights[i]
        return jnp.where(i == j, probs / wi * (wi - 1), probs)
    return jnp.where(i == j, 0.0, probs)


@eqx.filter_jit
def _pairs_expected_probs(
    pairs: NodePairView,
    *args: Any,
    full_shape: bool = False,
    dequantize: bool = True,
    **kwargs: Any,
) -> jnp.ndarray:
    """Compute expected pairwise connection probabilities."""
    init_kwargs, kwargs = split_kwargs_by_signature(EdgeProbabilityIntegral, **kwargs)
    integrator = EdgeProbabilityIntegral.from_pairs(pairs, **init_kwargs)
    integral = integrator.integrate(*args, **kwargs)[0]
    if pairs.model.is_homogeneous and full_shape:
        n = pairs.model.n_units
        p = jnp.full(n * (n - 1) // 2, integral)
        return squareform(p)
    if pairs.model.is_homogeneous or not pairs.model.is_quantized or not dequantize:
        return integral
    # Dequantize branch
    errmsg = "Dequantization for expected probabilities is not implemented yet."
    raise NotImplementedError(errmsg)


@eqx.filter_jit
def _pairs_edge_distance_density(
    pairs: NodePairView,
    g: jnp.ndarray,
    *args: Any,
    **kwargs: Any,
) -> jnp.ndarray:
    """Compute edge distance density."""
    pairs_kwargs, kwargs = split_kwargs_by_signature(pairs.probs, **kwargs)
    expected_probs = pairs.expected_probs(*args, **kwargs)
    probs = pairs.probs(g, **pairs_kwargs)
    gdensity = pairs.model.manifold.distance_density(g)
    return jnp.where(expected_probs == 0, 0.0, probs * gdensity / expected_probs)
