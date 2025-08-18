import math
from collections.abc import Iterator, Mapping, Sequence
from functools import singledispatchmethod
from typing import Any, NamedTuple, Self

import igraph as ig
import numpy as np
from scipy.integrate import quad
from scipy.sparse import csr_array, sparray
from scipy.special import expit

from . import options
from .kernels import AbstractGeometricKernel
from .manifolds import Sphere
from .optim import KBarOptimizer


class GRGGSample(NamedTuple):
    """Sample from the GRGG model.

    Attributes
    ----------
    A
        Sparse adjacency matrix of the sampled graph.
    X
        Coordinates of the sampled points on the sphere.
    G
        :mod:`igraph` representation of the sampled graph.
    """

    A: sparray
    X: np.ndarray

    @property
    def G(self) -> ig.Graph:
        """Return the :mod:`igraph` representation of the sampled graph."""
        # Make igraph graph from sparse adjacency matrix
        edges = np.column_stack(self.A.nonzero())
        G = ig.Graph(edges, directed=False, n=self.A.shape[0])
        return G.simplify()


class GRGG:
    """Generalized Random Geometric Graph Model.

    Attributes
    ----------
    n_nodes
        Number of nodes in the graph.
        The number of nodes determines also the surface area of the sphere,
        which is always equal to `n_nodes`, resulting in the sampling density of 1.
    manifold
        Manifold on which the graph is defined.
        Currently only :class:`grgg.manifolds.Sphere` is supported.
        Note that the manifold itself is always in the canonical form
        (e.g. unit sphere), and the rescaling to the desired radius and surface area
        is handled by the model.
        If an integer is passed, it is interpreted as the surface dimension
        of a sphere.
    kernels
        List of kernel functions defining the edge probabilities.

    Examples
    --------
    The model is initialized from a number of nodes and a manifold
    on which they will live (currently only spheres are supported).

    >>> from math import isclose
    >>> from grgg import GRGG, Sphere, Similarity, Complementarity
    >>> sphere = Sphere(2)  # sphere with 2-dimensional surface
    >>> model = GRGG(100, sphere)

    Alternatively, a model can be initialized by passing an integer
    specifying the surface dimension of a sphere instead of a full
    manifold instance.

    >>> model = GRGG(100, 2)

    Importantly, the manifold itself is always in the canonical form
    (e.g. unit sphere), and the rescaling to the desired radius and surface area
    is handled by the model.

    >>> model.manifold.radius()
    1.0

    Note that the model keeps track of the radius necessary
    for obtaining the desired surface area.
    >>> isclose(model.radius, model.manifold.radius(model.n_nodes))
    True
    >>> isclose(model.manifold.surface_area(model.radius), model.n_nodes)
    True

    For the model to be useful, we need to add kernel functions
    that define the edge probabilities. This can be done using the `add_kernel`
    method. The kernel functions must inherit from `AbstractGeometricKernel`.

    Below is an example of how to add a `Similarity` kernel with default parameters.
    In practice, this is not very useful as this way it is hard to control
    the average degree of the graph. The kernel is added in place, but the method also
    returns the referecnce to the model itself, so multiple invocations can be chained.
    >>> model.add_kernel(Similarity)
    GRGG(100, Sphere(2), Similarity(mu=..., beta=3.0, logspace=True))

    Note the 'logspace=True' parameter, which is the default for all kernels.
    This means that the kernel will use logarithmic distance-relations to allow for
    small-world effects. If you want to disable this behavior, you can pass
    'logspace=False' to the kernel constructor, or set it globally
    using the `options` module.

    >>> from grgg import options
    >>> options.logspace = False  # disable logarithmic distance for all kernels
    >>> model.add_kernel(Similarity)
    GRGG(100, Sphere(2), Similarity(mu=..., beta=3.0, logspace=False))
    >>> options.logspace = True  # restore default behavior

    Now, it is typically more useful to add kernels with specific average degrees.
    This is done by passing the desired average degree as the first argument
    to the `add_kernel` method. The kernel will be calibrated to induce the desired
    average degree in the graph.
    >>> model = GRGG(100, 2).add_kernel(5, Similarity)
    >>> isclose(model.kbar, 5.0, rel_tol=1e-4)
    True

    Most importantly, the model can have multiple kernels,
    which allows for more complex edge probability distributions.
    For example, we can add a `Complementarity` kernel with the same average degree.

    >>> model = model.add_kernel(5, Complementarity)

    This also gives us a good oportunity to note that, given a compound model,
    it is possible to get submodels with selected kernels using indexing.
    This allows us to see that the submodels indeed have the target average degree.
    >>> isclose(model[0].kbar, 5.0, rel_tol=1e-4)  # Similarity kernel
    True
    >>> isclose(model[1].kbar, 5.0, rel_tol=1e-4)  # Complementarity kernel
    True

    However, due to possible overlaps of the connections defined by different kernels,
    the average degree of the combined model may be lower than the sum of the
    average degrees of the submodels.
    >>> model.kbar < 10
    True

    To address this issue, the model provides a `calibrate` method,
    which allows to adjust the average degree of the model to a desired value.
    The method takes the desired average degree as the first argument,
    and an optional `weights` argument, which allows to set the relative weights
    of the kernels in the model. If not provided, all kernels are treated equally.

    >>> model.calibrate(10).kbar  # calibrate the model to average degree of 10
    10.0

    Moreover, it is possible to calibrate while assuming different kernel strengths.
    This can be done easily by passing a second 'weights' argument to the 'calibrate'
    method.

    Below we calibrate to 'kbar=10' while assuming that the second kernel
    (Complementarity) is twice as strong as the first one (Similarity).
    >>> model = model.calibrate(10, [1, 2])
    >>> model.kbar
    10.0
    >>> model[0].kbar
    3.3389203
    >>> model[1].kbar
    6.6776256
    """

    def __init__(
        self,
        n_nodes: int,
        manifold: Sphere | int,
        *kernels: AbstractGeometricKernel,
    ) -> None:
        if n_nodes <= 0:
            errmsg = "'n_nodes' must be positive"
            raise ValueError(errmsg)
        if isinstance(manifold, int):
            manifold = Sphere(manifold)
        if not isinstance(manifold, Sphere):
            errmsg = "'manifold' must be an instance of Sphere"
            raise TypeError(errmsg)
        if not all(isinstance(k, AbstractGeometricKernel) for k in kernels):
            errmsg = "'kernels' must inherit from 'AbstractGeometricKernel'"
            raise TypeError(errmsg)
        self.n_nodes = n_nodes
        self.manifold = manifold
        self.kernels = list(kernels)

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

    def copy(self) -> Self:
        """Create a copy of the GRGG model with optional modifications."""
        return self.__copy__()

    @property
    def radius(self) -> float:
        return self.manifold.radius(self.n_nodes)

    @property
    def submodels(self) -> Iterator[Self]:
        for i in range(len(self.kernels)):
            yield self[i]

    @property
    def kbar(self) -> float:
        """Average degree of the graph."""
        R = self.radius
        eps = np.mean([k.eps for k in self.kernels])

        def integrand(d: float) -> float:
            r = R * math.sin(d / R)
            S = self.manifold.surface_area(r, self.manifold.dim)
            return self.edgeprobs(d) * S

        integral, _ = quad(integrand, eps, R * np.pi)
        return integral / self.n_nodes * (self.n_nodes - 1)

    def edgeprobs(self, d: float | np.ndarray) -> float:
        """Probability of connection between two points at distance `d`."""
        if not self.kernels:
            errmsg = "At least one kernel function must be defined."
            raise ValueError(errmsg)
        P = None
        for kernel in self.kernels:
            K = kernel(d)
            p = 1 - expit(-K)
            if P is None:
                P = p
            else:
                P *= p
        if np.isscalar(d):
            P = P.item()
        return 1 - P  # type: ignore

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
        >>> from grgg import GRGG, Sphere, Similarity, Complementarity
        >>> GRGG(100, Sphere(2)).add_kernel(Similarity)
        GRGG(100, Sphere(2), Similarity(mu=..., beta=3.0, logspace=True))

        However, typically it is more useful to add a kernels inducing specific
        average degrees. This can be done by passing the desired average degree
        as the first argument to the `add_kernel` method.

        Below we define a model with two kernels, both inducing an average degree of 5.

        >>> model = (
        ...     GRGG(100, Sphere(2))
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
        kernel = kernel_type.from_manifold(self.manifold, self.n_nodes, **kwargs)
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
        kernel = kernel_type.from_manifold(self.manifold, self.n_nodes, **kwargs)
        model = self.copy()
        model.kernels = [kernel]
        optim = optim or {}
        mu = model._solve_for_kbar(kbar, **optim)
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
        *,
        optim: Mapping | None = None,
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
        optim
            Optional optimization parameters for the
            :func:`scipy.optimize.minimize` function.

        Examples
        --------
        The model can be calibrated to have a specific average degree
        by calling the `calibrate` method with the desired average degree.
        This method ensures that the overall average degree of the model
        is equal to the specified `kbar`, taking into account the possible overlaps
        of connections defined by different kernels.

        >>> from grgg import GRGG, Sphere, Similarity, Complementarity
        >>> model = (
        ...     GRGG(100, Sphere(2))
        ...     .add_kernel(Similarity)
        ...     .add_kernel(Complementarity)
        ...     .calibrate(10)
        ... )
        >>> model.kbar
        10.0

        It is also possible to calibrate the model while assuming different relative
        strengths of the kernels. This can be done by passing a second `weights`
        argument to the `calibrate` method.

        Below we calibrate to `kbar=10` while assuming that the second kernel
        (Complementarity) is twice as strong as the first one (Similarity).

        >>> model = model.calibrate(10, [1, 2])
        >>> model.kbar
        10.0
        >>> model[0].kbar  # Similarity kernel
        3.3389029
        >>> model[1].kbar  # Complementarity kernel
        6.6776256
        """
        optim = optim or {}

        mu = self._solve_for_kbar(kbar, weights, **optim)
        self.set_mu(mu)
        return self

    def set_mu(self, mu: Sequence) -> None:
        """Set the `mu` parameter for each kernel in the model."""
        for m, kernel in zip(mu, self.kernels, strict=True):
            if m is not None:
                kernel.mu = float(m)

    def sample(
        self,
        *,
        batch_size: int | None = None,
        random_state: np.random.Generator | int | None = None,
    ) -> GRGGSample:
        """Sample a graph from the GRGG model."""
        batch_size = options.sample_batch_size if batch_size is None else batch_size
        batch_size = int(batch_size)
        if batch_size <= 0:
            errmsg = "'batch_size' must be positive"
            raise ValueError(errmsg)
        if random_state is None:
            random_state = np.random.default_rng()
        elif isinstance(random_state, int):
            random_state = np.random.default_rng(random_state)
        if not isinstance(random_state, np.random.Generator):
            errmsg = "'random_state' must be a numpy random generator or an integer"
            raise TypeError(errmsg)
        n = self.n_nodes
        r = self.radius
        # Sample normalized unit sphere positions
        X = self.manifold.random_uniform(n)
        Ai = []
        Aj = []
        for i in range(0, n, batch_size):
            for j in range(0, i + batch_size, batch_size):
                D = self.manifold.cdist(X[i : i + batch_size], X[j : j + batch_size])
                P = self.edgeprobs(D * r)
                if i == j:
                    P = np.tril(P, k=-1)
                ai, aj = np.nonzero(random_state.random(P.shape) < P)
                ai += i * batch_size
                aj += j * batch_size
                Ai.extend(ai)
                Aj.extend(aj)
        values = np.ones(len(Ai), dtype=int)
        A = csr_array((values, (Ai, Aj)), shape=(n, n))
        A += A.T  # make it symmetric
        return GRGGSample(A, X * r)

    # Internals ----------------------------------------------------------------------

    def _solve_for_kbar(
        self,
        kbar: float,
        weights: np.ndarray | None = None,
        *,
        x0: np.ndarray | None = None,
        method: str = "Nelder-Mead",
        **kwargs: Any,
    ) -> np.ndarray:
        """Estimate the `mu` parameter from the desired average degree `kbar`."""
        optimizer = KBarOptimizer(self.copy(), kbar, weights)
        return optimizer.optimize(x0, method, **kwargs)


# Run doctests not discoverable in the standard way due to decorator usage -----------
__test__ = {
    "GRGG.add_kernel": GRGG.add_kernel.__doc__,
}
