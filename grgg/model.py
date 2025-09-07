import math
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property, singledispatchmethod
from typing import Any, Self

import igraph as ig
import numpy as np
from pathcensus import PathCensus
from scipy.sparse import csr_array, sparray
from tqdm.auto import tqdm

from . import options
from .integrate import Integration
from .layers import AbstractGRGGLayer
from .manifolds import CompactManifold, Sphere
from .quantize import ArrayQuantizer

__all__ = ("GRGG",)


class GRGG:
    """Generalized Random Geometric Graph.

    Attributes
    ----------
    n_nodes
        Number of nodes in the ensemble.
    manifold
        Compact isotropic manifold where nodes are embedded.
        Currently supported manifolds are:
        - :class:`~grgg.manifolds.Sphere`
        Can also be specified as an integer, which is interpreted as
        the dimension of a sphere with volume equal to `n_nodes`.
        Note that in this context manifold volume refers to the volume
        of the manifold surface, not of the enclosed space.
    layers
        List of GRGG layers with different energy and coupling functions.

    Examples
    --------
    Create a GRGG model with 100 nodes on a 2-sphere and a single
    similarity layer with default parameters.
    >>> from grgg import GRGG, Similarity
    >>> model = GRGG(100, 2, Similarity())
    >>> model
    GRGG(100, Sphere(2, r=2.82), Similarity(Beta(1.50), Mu(0.00), log=True))
    """

    def __init__(
        self,
        n_nodes: int,
        manifold: CompactManifold | int | tuple[int, type[CompactManifold]],
        *layers: AbstractGRGGLayer,
    ) -> None:
        """Initialize the GRGG model.

        Examples
        --------
        Initialize core model with 100 in 2D. The default manifold is a sphere.
        Moreover, by default the volume of the manifold is set to `n_nodes`.
        This leads to the unitary sampling density.
        >>> from grgg import GRGG, Sphere, Similarity, Complementarity
        >>> model = GRGG(100, 2)
        >>> model.manifold
        Sphere(2, r=2.82)
        >>> model.manifold.volume
        100.0
        >>> model.delta  # unitary sampling density
        1.0

        However, a model can be initialized also more explicitly from a sphere with
        arbitrary radius.
        >>> model = GRGG(100, Sphere(2, r=10))
        >>> model.manifold
        Sphere(2, r=10.00)
        >>> model.manifold.volume
        1256.637061
        >>> model.delta
        0.0795774715

        In order to pass an arbitrary manifold in `d` dimensions
        (but note that currently only spheres are supported), which then is tuned
        to have volume equal to `n_nodes`, one use the following syntax:
        >>> GRGG(100, Sphere, 2)
        GRGG(100, Sphere(2, r=2.82))
        >>> GRGG(100, 2, Sphere)
        GRGG(100, Sphere(2, r=2.82))

        Layers can also be defined at initialization.
        >>> GRGG(100, 2, Similarity, Complementarity(beta=10))
        GRGG(100, Sphere(...), Similarity(...), Complementarity(Beta(10.00), ...))
        """
        self._n_nodes = int(n_nodes)
        self.quantizer = ArrayQuantizer()
        # Handle manifold initialization
        if layers and (
            isinstance(layers[0], int)
            or isinstance(layers[0], type)
            and not issubclass(layers[0], AbstractGRGGLayer)
        ):
            dim = layers[0]
            layers = layers[1:]
            self.manifold = self._make_manifold((dim, manifold))
        else:
            self.manifold = self._make_manifold(manifold)
        self.layers = ()
        for layer in layers:
            self.add_layer(layer)
        # Initialize integration and optimization namespaces
        self.integrate = Integration(self)

    def __repr__(self) -> str:
        params = f"{self.n_nodes}, {self.manifold}"
        if self.layers:
            params += ", " + ", ".join(repr(layer) for layer in self.layers)
        return f"{self.__class__.__name__}({params})"

    def __copy__(self) -> Self:
        layers = [layer.copy() for layer in self.layers]
        return self.__class__(self.n_nodes, self.manifold.copy(), *layers)

    def __getitem__(self, index: int | slice) -> Self:
        layers = self.layers[index]
        if isinstance(layers, AbstractGRGGLayer):
            layers = [layers]
        return self.__class__(self.n_nodes, self.manifold, *layers)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GRGG):
            return NotImplemented
        return (
            self.n_nodes == other.n_nodes
            and self.manifold == other.manifold
            and self.layers == other.layers
        )

    def __hash__(self) -> int:
        return hash((self.n_nodes, self.manifold, self.layers))

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the ensemble."""
        if self.is_quantized:
            return len(self.quantizer.bins)
        return self._n_nodes

    @property
    def is_quantized(self) -> bool:
        """Whether the model parameters are quantized."""
        return not self.quantizer.is_empty

    @property
    def delta(self) -> float:
        """Sampling density."""
        return self._n_nodes / self.manifold.volume

    @property
    def submodels(self) -> Iterator[Self]:
        """Iterate over single-layer submodels.

        Examples
        --------
        >>> from grgg import GRGG, Similarity, Complementarity
        >>> model = GRGG(100, 2, Similarity, Complementarity)
        >>> submodels = list(model.submodels)
        >>> len(submodels)
        2
        >>> submodels[0]
        GRGG(100, Sphere(2, r=2.82), Similarity(Beta(1.50), Mu(0.00), log=True))
        >>> submodels[1]
        GRGG(100, Sphere(2, r=2.82), Complementarity(Beta(1.50), Mu(0.00), log=True))
        """
        for i in range(len(self.layers)):
            yield self[i]

    @property
    def is_heterogeneous(self) -> bool:
        """Whether the model is heterogeneous."""
        return any(layer.is_heterogeneous for layer in self.layers)

    @property
    def degree(self) -> np.ndarray:
        if self.is_heterogeneous:
            return np.array([self.integrate.degree(i)[0] for i in range(self.n_nodes)])
        return np.full(self.n_nodes, self.kbar)

    @property
    def kbar(self) -> float:
        r"""Average degree :math:`\bar{k}` of the GRGG model."""
        if self.is_heterogeneous:
            return self.degree.mean()
        return self.integrate.degree()[0]

    @property
    def density(self) -> float:
        """Expected density of the GRGG model."""
        return self.kbar / (self.n_nodes - 1)

    def dist2prob(self, g: np.ndarray, *args: Any) -> np.ndarray:
        """Convert distances to connection probabilities.

        Parameters
        ----------
        g
            Geodesic distances.
        *args
            Optional indices of the nodes corresponding to the distances `g`.
            If not provided it is assumed that `g` contains all pairwise distances.

        Examples
        --------
        >>> from grgg import GRGG, Similarity, Complementarity
        >>> model = GRGG(100, 2, Similarity())
        >>> d = np.array([0, 1, 2])
        >>> model.dist2prob(d)
        array([1.        , 0.5       , 0.11111111])

        Check that complementairity is maximal at the diameter of the manifold.
        >>> model = GRGG(100, 2, Complementarity())
        >>> model.dist2prob(model.manifold.diameter)
        1.0

        Check maximal probabilities in the joint similarity-complementarity model.
        >>> model = GRGG(100, 2, Similarity(), Complementarity())
        >>> model.dist2prob([0, model.manifold.diameter])
        array([1., 1.])

        Note that the maximum does not have to be 1 if the log-energies are not used.
        >>> GRGG(100, 1, Similarity(mu=3, log=False)).dist2prob(0)
        0.944092325269241
        >>> model = GRGG(100, 1, Complementarity(mu=3, log=False))
        >>> model.dist2prob(model.manifold.diameter)
        0.944092325269241
        """
        if not self.layers:
            errmsg = "the model has no layers"
            raise AttributeError(errmsg)
        if not np.isscalar(g):
            g = np.asarray(g)
        P = None
        for layer in self.layers:
            couplings = layer.coupling(g, *args)
            p = layer(couplings)
            P = 1 - p if P is None else P * (1 - p)
        if np.isscalar(P) or P.size == 1:
            P = P.item()
        return 1 - P  # type: ignore

    def pmatrix(self, full: bool = True, **kwargs: Any) -> np.ndarray:
        """Compute the full matrix of connection probabilities.

        Parameters
        ----------
        full
            Whether to return the full matrix or only the lower triangle.
        **kwargs
            Passed to :meth:`~grgg.manifolds.CompactManifold.sample_points`

        Examples
        --------
        >>> from grgg import GRGG, Similarity
        >>> model = GRGG(10, 2, Similarity())
        >>> P = model.pmatrix()
        >>> P.shape
        (10, 10)
        >>> bool(np.all(P >= 0) and np.all(P <= 1))
        True
        >>> bool(np.allclose(P, P.T))
        True
        >>> bool(np.all(np.diag(P) == 0))
        True

        Only the lower triangle can be computed to save memory.
        >>> P = model.pmatrix(full=False)
        >>> P.shape
        (45,)
        """
        n = self.n_nodes
        X = self.manifold.sample_points(n, **kwargs)
        D = self.manifold.pdist(X, full=full)
        P = self.dist2prob(D)
        if full:
            np.fill_diagonal(P, 0)
        return P

    def sample(
        self,
        *,
        batch_size: int | None = None,
        random_state: int | np.random.Generator | None = None,
        progress: bool | None = None,
    ) -> "GRGGSample":
        """Sample graph instance from the GRGG ensemble.

        Parameters
        ----------
        batch_size
            Number of nodes to sample in one batch. If non-positive, all nodes are
            sampled in one batch. Batching can be useful to reduce memory consumption
            for large graphs, at the cost of a somewhat longer runtime.
        random_state
            Random state for reproducibility.
            If not provided, a new random state is created from system entropy.
        progress
            Whether to display a progress bar.

        Returns
        -------
        GRGGSample
            A named tuple containing the adjacency matrix, coordinates of the sampled
            points, and the igraph representation of the sampled graph.

        Examples
        --------
        Sample a graph from a similarity GRGG model.
        >>> from grgg import GRGG, Similarity
        >>> model = GRGG(1000, 2, Similarity())
        >>> sample = model.sample(random_state=42)
        >>> sample.G.transitivity_undirected() > 0.1
        True
        """
        n_nodes = self.n_nodes
        batch_size = options.sample.batch_size if batch_size is None else batch_size
        batch_size = int(batch_size)
        if batch_size <= 0:
            batch_size = n_nodes
        if not isinstance(random_state, np.random.Generator):
            random_state = np.random.default_rng(random_state)
        X = self.manifold.sample_points(n_nodes, random_state=random_state)
        Ai = []
        Aj = []
        # Sample edges in batches to avoid memory issues with large graphs
        # consider only the lower triangle of the adjacency matrix
        # as the graph is undirected
        nb = int(math.ceil(n_nodes / batch_size))
        n_batches = nb * (nb + 1) // 2
        if progress is None:
            progress = n_batches >= options.sample.auto_progress
        pbar = tqdm(
            total=n_batches, disable=not progress, desc="Sampling", unit=" batches"
        )
        for i in range(0, n_nodes, batch_size):
            xi = slice(i, i + batch_size)
            for j in range(0, i + batch_size, batch_size):
                if i == j:
                    ai, aj = self._sample_diag(X, xi, random_state)
                else:
                    yi = slice(j, j + batch_size)
                    ai, aj = self._sample_offdiag(X, xi, yi, random_state)
                ai += i
                aj += j
                Ai.append(ai)
                Aj.append(aj)
                pbar.update(1)
        pbar.close()
        Ai = np.concatenate(Ai)
        Aj = np.concatenate(Aj)
        values = np.ones(len(Ai), dtype=int)
        A = csr_array((values, (Ai, Aj)), shape=(n_nodes, n_nodes))
        A += A.T  # make it symmetric
        return GRGGSample(A, X)

    def add_layer(
        self,
        layer: AbstractGRGGLayer,
        # **constraints: float | np.ndarray,
    ) -> Self:
        """Add a layer to the GRGG model.

        Parameters
        ----------
        layer
            A GRGG layer to add.
        **constraints
            Soft constraints on the sufficient statistics.

        Examples
        --------
        Add a default complementarity layer and check that it is linked to the model.
        >>> from grgg import GRGG, Complementarity
        >>> model = GRGG(100, 1).add_layer(Complementarity())
        >>> model.layers[0].model is model
        True
        """
        if isinstance(layer, type) and issubclass(layer, AbstractGRGGLayer):
            layer = layer()
        if not isinstance(layer, AbstractGRGGLayer):
            errmsg = "'layer' must be a 'AbstractGRGGLayer' instance"
            raise TypeError(errmsg)
        layer.model = self
        self.layers = (*self.layers, layer)
        return self

    def remove_layer(self, index: int) -> Self:
        """Remove a layer from the GRGG model.

        Parameters
        ----------
        index
            Index of the layer to remove.
        """
        self.layers = tuple(layer for i, layer in enumerate(self.layers) if i != index)
        return self

    def copy(self) -> Self:
        return self.__copy__()

    def make_idx(
        self, g: np.ndarray, *idx: slice | np.ndarray
    ) -> np.ndarray | tuple[np.ndarray, ...]:
        """Make index tuple for array indexing."""
        if not idx and not np.isscalar(g):
            if g.ndim == 1:
                n = self.n_nodes
                A = np.tril(np.ones((n, n), dtype=bool), k=-1)
                idx = np.column_stack(np.nonzero(A))  # type: ignore
            else:
                idx = (slice(None), slice(None))
        return idx

    def quantize(self, **kwargs: Any) -> Self:
        """Quantize node parameters.

        `**kwargs` are passed to :class:`~grgg.quantize.KMeansQuantizer`.
        """
        if self.is_quantized:
            return self
        self.quantizer.discretizer.set_params(**kwargs)
        # Collect parameters
        params = []
        for layer in self.layers:
            for param in (layer.beta, layer.mu):
                if param.heterogeneous:
                    params.append(param.values)  # noqa
        if not params:
            return self
        params = np.column_stack(params)
        # Fit quantizer and quantize parameters
        quantized = self.quantizer.quantize(params)
        i = 0
        for layer in self.layers:
            for param in (layer.beta, layer.mu):
                if param.heterogeneous:
                    param.value = quantized[:, i]
                    i += 1
        return self

    def dequantize(self) -> Self:
        """Remove quantization of node parameters."""
        if self.is_quantized:
            params = self.quantizer.dequantize(clear=True)
            i = 0
            for layer in self.layers:
                for param in (layer.beta, layer.mu):
                    if param.heterogeneous:
                        param.value = params[:, i]
                        i += 1
        return self

    # Internals ----------------------------------------------------------------------

    def _sample_diag(
        self, X: np.ndarray, xi: slice, random_state: np.random.Generator
    ) -> np.ndarray:
        """Sample edges from a diagonal batch of the probability matrix."""
        x = X[xi]
        n = len(x)
        D = self.manifold.pdist(x, full=False)
        P = self.dist2prob(D, xi)
        M = random_state.random(P.shape) < P
        A = np.zeros((n, n), dtype=bool)
        A[np.triu_indices_from(A, k=1)] = M
        ai, aj = np.nonzero(A.T)
        return ai, aj

    def _sample_offdiag(
        self, X: np.ndarray, xi: slice, yi: slice, random_state: np.random.Generator
    ) -> np.ndarray:
        """Sample edges from an off-diagonal batch of the probability matrix."""
        D = self.manifold.cdist(X[xi], X[yi])
        P = self.dist2prob(D, xi, yi)
        M = random_state.random(P.shape) < P
        ai, aj = np.nonzero(M)
        return ai, aj

    @singledispatchmethod
    def _make_manifold(self, manifold: CompactManifold) -> CompactManifold:
        if isinstance(manifold, tuple):
            manifold_type, dim = manifold
            if isinstance(manifold_type, int):
                manifold_type, dim = dim, manifold_type
            return self._make_manifold(dim, manifold_type)
        if not isinstance(manifold, CompactManifold):
            errmsg = "'manifold' must be a 'CompactManifold'"
            raise TypeError(errmsg)
        return manifold

    @_make_manifold.register
    def _(
        self, dim: int, manifold_type: type[CompactManifold] = Sphere
    ) -> CompactManifold:
        manifold = manifold_type(dim).with_volume(self.n_nodes)
        return self._make_manifold(manifold)

    @_make_manifold.register
    def _(self, dim: np.integer, *args: Any, **kwargs: Any) -> CompactManifold:
        return self._make_manifold(int(dim), *args, **kwargs)


@dataclass(frozen=True)
class GRGGSample:
    """Sample from the GRGG model.

    Attributes
    ----------
    A
        Sparse adjacency matrix of the sampled graph.
    X
        Coordinates of the sampled points on the sphere.
    G
        :mod:`igraph` representation of the sampled graph.
    census
        Path census of the sampled graph (see :mod:`pathcensus`).
        It allows for efficient computation of comprehensive structural
        similarity and complementarity coefficients.
    """

    A: sparray
    X: np.ndarray

    @cached_property
    def G(self) -> ig.Graph:
        """Return the :mod:`igraph` representation of the sampled graph."""
        # Make igraph graph from sparse adjacency matrix
        edges = np.column_stack(self.A.nonzero())
        G = ig.Graph(edges, directed=False, n=self.A.shape[0])
        return G.simplify()

    @cached_property
    def struct(self) -> PathCensus:
        """Return the path census for calculating structural coefficients.

        See :mod:`pathcensus` for details.
        """
        return PathCensus(self.A)
