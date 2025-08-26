from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import cached_property, singledispatchmethod
from typing import Any, Self

import igraph as ig
import numpy as np
from pathcensus import PathCensus
from scipy.sparse import csr_array, sparray
from scipy.special import expit

from . import options
from .integrate import GRGGIntegration
from .kernels import AbstractGeometricKernel
from .manifolds import CompactManifold, Manifold, Sphere
from .optimize import GRGGOptimization


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
    def census(self) -> PathCensus:
        """Return the path census of the sampled graph."""
        return PathCensus(self.A)


class GRGG:
    """Generalized Random Geometric Graph Model.

    Attributes
    ----------
    n_nodes
        Number of nodes in the graph.
    manifold
        compact manigold on which the graph is to be defined.
        Currently only :class:`grgg.manifolds.Sphere` is supported.
        Can also be specified as an integer, which will be interpreted
        as the surface dimension of a sphere with surface area equal to `n_nodes`.
    kernels
        List of kernel functions defining the edge probabilities.
    rho
        Sampling density, which is the ratio of the number of nodes
        to the surface area of the manifold.

    Examples
    --------
    The model is initialized from a number of nodes and a manifold
    on which they will live (currently only spheres are supported).

    >>> from math import isclose
    >>> from grgg import GRGG, Sphere, Similarity, Complementarity
    >>> sphere = Sphere(2)  # unit sphere with 2-dimensional surface
    >>> model = GRGG(100, sphere)

    Note that in this case the sampling density, `rho`, is not equal to 1.
    >>> model.rho != 1
    True

    Alternatively, a model can be initialized by passing an integer
    specifying the surface dimension of a sphere instead of a full
    manifold instance. In this case, it is assumed that the sphere
    has a surface area equal to the number of nodes, so the sampling density
    is always 1. This is the most typical way to initialize the model.
    >>> model = GRGG(100, 2)
    >>> model.rho
    1.0

    For the model to be useful, we need to add kernel functions that define the edge
    probabilities. This can be done using the `add_kernel` method. The kernel functions
    must inherit from :class:`grgg.kernels.AbstractGeometricKernel`.

    Below is an example of how to add a :class:`grgg.kernels.Similarity` kernel with
    default parameters. The kernel is added in place, but the method also returns the
    reference to the model itself, so multiple invocations can be chained.
    >>> model.add_kernel(Similarity)
    GRGG(100, Sphere(...), Similarity(..., logspace=False))

    Note the 'logspace=False' parameter, which is the default for all kernels.
    This means that the kernel will NOT USE logarithmic distance-relations to allow for
    small-world effects. If you want to change this behavior, you can pass
    'logspace=True' to the kernel constructor, or set it globally using the `options`
    module.

    >>> from grgg import options
    >>> options.kernel.logspace = True  # enable logarithmic distance for all kernels
    >>> model.add_kernel(Similarity)
    GRGG(100, Sphere(2, r=...), Similarity(mu=..., beta=3.0, logspace=True))
    >>> options.kernel.logspace = False  # restore default behavior

    Temporary options handling can be done more conveniently using the context manager.
    >>> with options:
    ...     options.kernel.logspace = True
    ...     model.add_kernel(Similarity)
    GRGG(100, Sphere(2, r=...), Similarity(mu=..., beta=3.0, logspace=True))

    >>> options.kernel.logspace
    False

    Now, it is typically more useful to add kernels with specific average degrees.
    This is done by passing the desired average degree as the first argument to the
    :meth:`add_kernel` method. The kernel will be calibrated to induce the desired
    average degree in the graph.
    >>> model = GRGG(100, 2).add_kernel(5, Similarity)
    >>> isclose(model.kbar, 5.0, rel_tol=1e-4)
    True

    Most importantly, the model can have multiple kernels,
    which allows for more complex edge probability distributions.
    For example, we can add a :class:`grgg.kernels.Complementarity`
    kernel with the same average degree.

    >>> model = model.add_kernel(5, Complementarity)

    This also gives us a good oportunity to note that, given a compound model,
    it is possible to get submodels with selected kernels using indexing.
    This allows us to see that the submodels indeed have the target average degree.
    >>> isclose(model[0].kbar, 5.0, rel_tol=1e-4)  # Similarity kernel
    True
    >>> isclose(model[1].kbar, 5.0, rel_tol=1e-4)  # Complementarity kernel
    True

    However, due to possible overlaps of the connections defined by different kernels,
    the average degree of the combined model may be lower than the sum of the average
    degrees of the submodels.
    >>> model.kbar < 10
    True

    To address this issue, the model provides a `calibrate` method,  which allows
    for adjusting the average degree of the model to a desired value. The method takes
    the desired average degree as the first argument, and an optional `weights`
    argument, which allows for setting the relative weights of the kernels in the model.
    If not provided, all kernels are treated equally.

    >>> # calibrate the model to average degree of 10
    >>> isclose(10.0, model.calibrate(10).kbar, rel_tol=1e-4)
    True

    Note that in this case both kernels induce the same average degree.
    >>> isclose(model[0].kbar, model[1].kbar, rel_tol=1e-4)
    True

    Here we calibrate to `kbar=10` while assuming that the second kernel
    (Complementarity) is twice as strong as the first one (Similarity).
    >>> model = model.calibrate(10, [1, 2])
    >>> isclose(10.0, model.kbar, rel_tol=1e-4)
    True
    >>> isclose(model[0].kbar, 3.33, rel_tol=1e-2) # Similarity kernel
    True
    >>> isclose(model[1].kbar, 6.66, rel_tol=1e-2) # Complementarity kernel
    True
    """

    def __init__(
        self,
        n_nodes: int,
        manifold: CompactManifold | int | tuple[int, type[Manifold]],
        *kernels: AbstractGeometricKernel,
    ) -> None:
        # Check nodes specification
        if n_nodes <= 0:
            errmsg = "'n_nodes' must be positive"
            raise ValueError(errmsg)
        self.n_nodes = int(n_nodes)
        # Handle manifold initialization
        if kernels and isinstance(kernels[0], type | int):
            dim = kernels[0]
            kernels = kernels[1:]
            self.manifold = self._make_manifold((dim, manifold))
        else:
            self.manifold = self._make_manifold(manifold)
        # Check kernels
        if not all(isinstance(k, AbstractGeometricKernel) for k in kernels):
            errmsg = "'kernels' must inherit from 'AbstractGeometricKernel'"
            raise TypeError(errmsg)
        self.kernels = list(kernels)
        # Initialize integration and optimization namespaces
        self.integrate = GRGGIntegration(self)
        self.optimize = GRGGOptimization(self)

    @singledispatchmethod
    def _make_manifold(self, manifold: Manifold) -> Manifold:
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
    def _(self, dim: int, manifold_type: type[Manifold] = Sphere) -> Manifold:
        manifold = manifold_type.from_surface_area(dim, self.n_nodes)
        return self._make_manifold(manifold)

    @_make_manifold.register
    def _(self, dim: np.integer, *args: Any, **kwargs: Any) -> Manifold:
        return self._make_manifold(int(dim), *args, **kwargs)

    def __repr__(self) -> str:
        cn = self.__class__.__name__
        kernels = ", ".join(map(repr, self.kernels))
        attrs = f"{self.n_nodes}, {self.manifold!r}"
        if self.kernels:
            attrs += f", {kernels}"
        return f"{cn}({attrs})"

    def __copy__(self) -> Self:
        kernels = [k.copy() for k in self.kernels]
        return self.__class__(self.n_nodes, self.manifold.copy(), *kernels)

    def __getitem__(self, idx: int | slice) -> Self:
        """Get a copy of the GRGG model with a subset of kernels."""
        kernels = self.kernels[idx]
        if isinstance(idx, int):
            kernels = [kernels]
        return self.__class__(self.n_nodes, self.manifold, *kernels)

    def __call__(self, d: float | np.ndarray) -> float:
        """Evaluate the edge probabilities for distances `d`."""
        return self.edgeprobs(d)

    def copy(self) -> Self:
        """Create a copy of the GRGG model with optional modifications."""
        return self.__copy__()

    @property
    def rho(self) -> float:
        """Sampling density over the manifold."""
        return self.n_nodes / self.manifold.surface_area

    @property
    def submodels(self) -> Iterator[Self]:
        for i in range(len(self.kernels)):
            yield self[i]

    @property
    def kbar(self) -> float:
        """Average degree of the graph."""
        return self.integrate.kbar()

    @property
    def density(self) -> float:
        return self.kbar / (self.n_nodes - 1)

    def edgeprobs(self, d: float | np.ndarray) -> float:
        """Probability of connection between two points at distance `d`."""
        if not self.kernels:
            errmsg = "At least one kernel function must be defined."
            raise ValueError(errmsg)
        P = None
        for kernel in self.kernels:
            K = kernel(d)
            p = expit(-K)
            if P is None:
                P = 1 - p
            else:
                P *= 1 - p
        if np.isscalar(d):
            P = P.item()
        return 1 - P  # type: ignore

    def sample(
        self,
        *,
        batch_size: int | None = None,
        random_state: np.random.Generator | int | None = None,
    ) -> GRGGSample:
        """Sample a graph from the GRGG model.

        Parameters
        ----------
        batch_size
            Number of points to sample in each batch.
            If not provided, the default is 1000.
        random_state
            Random state or seed for reproducibility.
            If not provided, a new random state will be created.

        Returns
        -------
        GRGGSample
            A named tuple containing the adjacency matrix, coordinates of the sampled
            points, and the igraph representation of the sampled graph.

        Examples
        --------
        >>> from math import isclose
        >>> from grgg import GRGG, Similarity, Complementarity
        >>> from grgg.utils import random_generator
        >>> rng = random_generator(17089)
        >>> model = (
        ...     GRGG(100, 2)
        ...     .add_kernel(Similarity)
        ...     .add_kernel(Complementarity)
        ...     .calibrate(10)
        ... )

        Check that the average degree of the sampled graphs is close
        to the model expectation.
        >>> Kbars = [
        ...     model.sample(random_state=rng).A.sum(axis=1).mean()
        ...     for _ in range(100)
        ... ]
        >>> isclose(np.mean(Kbars), model.kbar, rel_tol=1e-2)
        True

        Check that the same holds when using batching.
        >>> Kbars = [
        ...     model.sample(batch_size=10, random_state=rng).A.sum(axis=1).mean()
        ...     for _ in range(100)
        ... ]
        >>> isclose(np.mean(Kbars), model.kbar, rel_tol=1e-2)
        True
        """
        batch_size = options.sample.batch_size if batch_size is None else batch_size
        batch_size = int(batch_size)
        if batch_size <= 0:
            errmsg = "'batch_size' must be positive"
            raise ValueError(errmsg)
        if not isinstance(random_state, np.random.Generator):
            random_state = np.random.default_rng(random_state)
        n_nodes = self.n_nodes
        X = self.manifold.sample_points(n_nodes, random_state=random_state)
        Ai = []
        Aj = []
        # Sample edges in batches to avoid memory issues with large graphs
        # consider only the lower triangle of the adjacency matrix
        # as the graph is undirected
        for i in range(0, n_nodes, batch_size):
            for j in range(0, i + batch_size, batch_size):
                if i == j:
                    x = X[i : i + batch_size]
                    n = len(x)
                    d = self.manifold.pdist(x, full=False)
                    D = np.zeros_like(d, shape=(n, n))
                    D[np.triu_indices_from(D, k=1)] = d
                    D = D.T
                else:
                    D = self.manifold.cdist(
                        X[i : i + batch_size],
                        X[j : j + batch_size],
                    )
                P = self.edgeprobs(D)
                if i == j:
                    idx = np.tril_indices_from(P, k=-1)
                    p = P[idx]  # type: ignore
                    a = np.zeros_like(P, dtype=bool)
                    a[idx] = random_state.random(p.shape) < p
                    ai, aj = np.nonzero(a)
                else:
                    ai, aj = np.nonzero(random_state.random(P.shape) < P)
                ai += i
                aj += j
                Ai.append(ai)
                Aj.append(aj)
        Ai = np.concatenate(Ai)
        Aj = np.concatenate(Aj)
        values = np.ones(len(Ai), dtype=int)
        A = csr_array((values, (Ai, Aj)), shape=(n_nodes, n_nodes))
        A += A.T  # make it symmetric
        return GRGGSample(A, X)

    @singledispatchmethod
    def add_kernel(
        self,
        kernel_type: type[AbstractGeometricKernel],
        *args: Any,  # noqa
        **kwargs: Any,
    ) -> Self:
        """Add a kernel function to the model.

        Parameters
        ----------
        [kbar]
            Optional first parameter to set the average node degree
            induced by the kernel.
        kernel_type : type[AbstractGeometricKernel]
            The kernel class to instantiate.
        **kwargs
            Additional parameters for the kernel.

        Examples
        --------
        Here we define a model with 100 nodes on a 2-dimensional sphere
        and add a `Similarity` with default parameters.

        >>> from math import isclose
        >>> from grgg import GRGG, Similarity, Complementarity
        >>> GRGG(100, 2).add_kernel(Similarity)
        GRGG(100, Sphere(...), Similarity(...))

        However, typically it is more useful to add a kernels inducing specific
        average degrees. This can be done by passing the desired average degree
        as the first argument to the `add_kernel` method.

        Below we define a model with two kernels, both inducing an average degree of 5.

        >>> model = (
        ...     GRGG(100, 2)
        ...     .add_kernel(5, Similarity)
        ...     .add_kernel(5, Complementarity)
        ... )
        >>> isclose(model[0].kbar, 5.0, rel_tol=1e-4)  # Similarity kernel submodel
        True
        >>> isclose(model[1].kbar, 5.0, rel_tol=1e-4)  # Complementarity kernel submodel
        True

        Note that the average degree of the combined model may be lower than the sum
        of the average degrees of the submodels due to overlaps in connections.
        >>> model.kbar < 10
        True

        To address this issue, the model provides a `calibrate` method.
        See :meth:`calibrate` for more details.
        """
        kernel = kernel_type.from_manifold(self.manifold, **kwargs)
        self.kernels.append(kernel)
        return self

    @add_kernel.register
    def _(
        self,
        kbar: float,
        kernel_type: type[AbstractGeometricKernel],
        *,
        optim: Mapping | None = None,
        **kwargs: Any,
    ) -> Self:
        kernel = kernel_type.from_manifold(self.manifold, **kwargs)
        model = self.copy()
        model.kernels = [kernel]
        optim = optim or {}
        mu = model.optimize.mu(kbar, **optim).x
        kernel.mu = float(mu[0])
        self.kernels.append(kernel)
        return self

    @add_kernel.register
    def _(self, kbar: int, *args: Any, **kwargs: Any) -> Self:
        return self.add_kernel(float(kbar), *args, **kwargs)

    @add_kernel.register
    def _(self, kbar: np.ndarray, *args: Any, **kwargs: Any) -> Self:
        if not np.isscalar(kbar):
            errmsg = "'kbar' must be a scalar"
            raise ValueError(errmsg)
        return self.add_kernel(float(kbar), *args, **kwargs)

    def calibrate(
        self,
        kbar: float,
        weights: np.ndarray | None = None,
        **kwargs: Any,
    ) -> Self:
        """Calibrate the model to have a specific average degree `kbar`.

        Parameters
        ----------
        kbar
            Desired average degree of the graph.
        weights
            Relative weights of the kernels in the model.
            If provided, it will be used to set the relative weights of the kernels.
            If not provided, all kernels will be treated equally.
        **kwargs
            Optional optimization parameters for the
            :func:`scipy.optimize.minimize` function.

        Examples
        --------
        The model can be calibrated to have a specific average degree
        by calling the `calibrate` method with the desired average degree.
        This method ensures that the overall average degree of the model
        is equal to the specified `kbar`, taking into account the possible overlaps
        of connections defined by different kernels.

        >>> from math import isclose
        >>> from grgg import GRGG, Similarity, Complementarity
        >>> model = (
        ...     GRGG(100, 2)
        ...     .add_kernel(Similarity)
        ...     .add_kernel(Complementarity)
        ...     .calibrate(10)
        ... )
        >>> isclose(10.0, model.kbar, rel_tol=1e-4)
        True

        It is also possible to calibrate the model while assuming different relative
        strengths of the kernels. This can be done by passing a second `weights`
        argument to the `calibrate` method.

        Below we calibrate to `kbar=10` while assuming that the second kernel
        (Complementarity) is twice as strong as the first one (Similarity).

        >>> model = model.calibrate(10, [1, 2])
        >>> isclose(10.0, model.kbar, rel_tol=1e-4)
        True
        >>> isclose(model[0].kbar, 3.33, rel_tol=1e-2)  # Similarity kernel
        True
        >>> isclose(model[1].kbar, 6.66, rel_tol=1e-2)  # Complementarity kernel
        True
        """
        mu = self.optimize.mu(kbar, weights=weights, **kwargs).x
        self.set_kernel_params(mu=mu)
        return self

    def set_kernel_params(self, **params: np.ndarray) -> Self:
        """Set kernel parameters."""
        for param, values in params.items():
            for kernel, value in zip(self.kernels, values, strict=True):
                setattr(kernel, param, value)
        return self


# Run doctests not discoverable in the standard way due to decorator usage -----------
__test__ = {
    "GRGG.add_kernel": GRGG.add_kernel.__doc__,
}
