import math
from collections.abc import MutableSequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Self

import igraph as ig
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.sparse import csr_array, issparse, sparray

from . import options
from .kernels import AbstractGeometricKernel
from .manifolds import Sphere
from .utils import (
    copy_with_update,
    get_rng,
    sphere_distances,
    sphere_surface_area,
    sphere_surface_sample,
)


class GRGGSample(NamedTuple):
    """Sample from the GRGG model.

    Attributes
    ----------
    A : np.ndarray
        Sparse adjacency matrix of the sampled graph.
    X : np.ndarray
        Coordinates of the sampled points on the sphere.
    G : ig.Graph
        :mod:`igraph` representation of the sampled graph.
    """

    A: sparray | np.ndarray
    X: np.ndarray

    @property
    def G(self) -> ig.Graph:
        """Return the :mod:`igraph` representation of the sampled graph."""
        if issparse(self.A):
            # Make igraph graph from sparse adjacency matrix
            edges = list(zip(*self.A.nonzero(), strict=True))
            return ig.Graph(edges, directed=False, n=self.A.shape[0])
        return ig.Graph.Adjacency(self.A, mode="undirected", loops=False)


@dataclass(init=False)
class GRGG:
    """Generalized Random Geometric Graph Model.

    Attributes
    ----------
    n : int
        Number of nodes in the graph.
    sphere : Sphere
        Sphere on which the graph is defined.
    kernel : AbstractGeometricKernel
        Kernel function defining the edge probabilities.
    logdist : bool, optional
        If `True`, the model uses logarithmic distances for edge probabilities.
    eps : float, optional
        Small value to avoid numerical issues with distances.

    Examples
    --------
    It is most convenient to create the model from the number of nodes `n`
    This method defined the surface area of the sphere to be equal to `n`,
    which means that the sampling density is equal to 1 over the surface area.

    >>> from grgg import GRGG, Similarity, Complementarity
    >>> rgg = GRGG.from_n(n=100, k=2)
    >>> rgg.sphere.S  # doctest: +FLOAT_CMP
    100.0

    Then, kernel functions defining the edge probabilities can be set
    using the `set_kernel` method. This also allows for calibrating
    the parameters of the kernel functions to fit particular average node degrees.
    >>> kbar = 10.0
    >>> rgg = rgg.set_kernel(Similarity, kbar=kbar, beta=3.0)
    >>> bool(np.isclose(rgg.kbar, kbar, atol=1e-1))
    True

    It is easy to sample from the model. Samples are represented as :class:`GRGGSample`
    instances, which contain the adjacency matrix, coordinates of the points
    on the sphere, and the :mod:`igraph` representation of the graph.
    >>> # 'sparse=True' is the default
    >>> # and produces 'scipy.sparse.csr_array' instances
    >>> sample = rgg.sample(sparse=False)

    Adjacency matrix is stored in the `A` attribute, and it is symmetric.
    >>> A = sample.A
    >>> bool(np.all(A == A.T))  # adjacency matrices are symmetric
    True

    Coordinates of the points on the sphere are stored in the `X` attribute.
    They are properly scaled to the sphere radius.
    >>> X = sample.X
    >>> bool(np.allclose(np.linalg.norm(X, axis=1), rgg.sphere.R))
    True

    And the :mod:`igraph` representation of the graph is available
    in the `G` attribute. It can be used for further analysis.
    >>> sample.G
    <igraph.Graph ...>

    Sampled adjacency matrices have the proper average degree in expectation
    if the model was calibrated correctly using the :meth:`GRGR.set_kernel` method.
    >>> K = np.array(
    ...     [rgg.sample(sparse=False).A.sum(axis=1).mean() for _ in range(1000)]
    ... )
    >>> bool(np.isclose(K.mean(), kbar, atol=1e-1))
    True

    All the above methods work exactly the same way when using :class:`Complementarity`
    or any other kernel function that inherits from :class:`AbstractGeometricKernel`.

    Moreover, it is possible to combine multiple kernel functions
    to create a more complex model such as the Similarity-Complementarity RGG model.
    In this case, the `set_kernel` method can be called multiple times,
    and the `calibrate` method can be used to set the average degree of the graph
    based on the combined effect of all kernels.
    >>> kbar = 10.0
    >>> rgg = (
    ...     GRGG.from_n(n=100, k=2)
    ...     .set_kernel(Similarity, kbar=kbar)
    ...     .set_kernel(Complementarity, kbar=kbar)
    ... )

    We can get submodels with selected kernels using indexing.
    This allows us to see that the submodels indeed have the target average degree.
    >>> bool(np.isclose(rgg[0].kbar, kbar, atol=1e-1))
    True
    >>> bool(np.isclose(rgg[1].kbar, kbar, atol=1e-1))
    True

    However, due to possible overlaps of the connections defined by different kernels,
    the average degree of the combined model may not be exactly equal to the sum of the
    average degrees of the submodels. This is why the `calibrate` method
    is used to set the average degree of the combined model. It accepts a `q` parameter
    that defines the relative weights of the kernels in the model.
    >>> q = 0.2  # relative weight of the similarity kernel
    >>> rgg = rgg.calibrate(kbar, q=q)
    >>> bool(np.isclose(rgg.kbar, kbar, atol=1e-1))
    True
    """

    n: int
    sphere: Sphere
    kernels: MutableSequence = field(default_factory=list)
    logdist: bool = options.logdist
    eps: float = options.eps

    def __init__(
        self,
        n: int,
        sphere: Sphere,
        *args: AbstractGeometricKernel,
        kernels: MutableSequence[AbstractGeometricKernel] = (),
        logdist: bool | None = None,
        eps: float | None = None,
    ) -> None:
        """Initialize the GRGG model.

        Kernels may be passed as `*args` and/or as `kernels` list.
        However, the two methods of passing kernels may not be combined.
        """
        if logdist is None:
            logdist = options.logdist
        if eps is None:
            eps = options.eps
        if n <= 0:
            errmsg = "'n' must be positive"
            raise ValueError(errmsg)
        if eps <= 0:  # type: ignore
            errmsg = "'eps' must be positive"
            raise ValueError(errmsg)
        if kernels and args:
            errmsg = "cannot combine 'kernels' list and '**args'"
            raise ValueError(errmsg)
        if args:
            kernels = args  # type: ignore
        self.n = n
        self.sphere = sphere
        self.kernels = list(kernels or [])
        self.logdist = logdist
        self.eps = eps

    def __copy__(self) -> Self:
        """Create a copy of the GRGG model."""
        return self.__class__(
            self.n,
            self.sphere,
            kernels=self.kernels,
            logdist=self.logdist,
            eps=self.eps,
        )

    def __getitem__(self, idx: int | slice) -> Self:
        """Get a copy of the GRGG model with a subset of kernels."""
        if isinstance(idx, slice):
            kernels = self.kernels[idx]
        elif isinstance(idx, int):
            kernels = [self.kernels[idx]]
        else:
            errmsg = "Index must be an integer or a slice"
            raise TypeError(errmsg)
        return self.__class__(
            self.n,
            self.sphere.copy(),
            kernels=[k.copy() for k in kernels],
            logdist=self.logdist,
            eps=self.eps,
        )

    @classmethod
    def from_n(cls, n: int, k: int, *args: Any, **kwargs: Any) -> Self:
        """Create an instance from the number of nodes.

        This method assumes that `S = n`, that is,
        the surface area of the sphere is equal to the number of nodes.

        Parameters
        ----------
        n : int
            Number of nodes in the graph.
        k : int
            Surface dimension of the sphere.
        *args, **kwargs : Any
            Additional arguments passed to the constructor.
        """
        sphere = Sphere.from_area(n, k)
        return cls(n, sphere, *args, **kwargs)

    def copy(self, **kwargs: Any) -> Self:
        """Create a copy of the GRGG model with optional modifications."""
        return copy_with_update(self, **kwargs)

    @property
    def rho(self) -> float:
        """Density of the graph, defined as `n / S`."""
        return self.n / self.sphere.S

    @property
    def kbar(self) -> float:
        """Average degree of the graph."""

        def integrand(d: float) -> float:
            R = self.sphere.R
            r = R * math.sin(d / R)
            S = sphere_surface_area(r, self.sphere.k - 1)
            return self.edgeprob(d) * self.rho * S

        integral, _ = quad(integrand, 0, self.sphere.R * np.pi)
        return integral / self.n * (self.n - 1)

    def edgeprob(self, d: float | np.ndarray) -> float:
        """Probability of connection between two points at distance `d`."""
        if not self.kernels:
            errmsg = "At least one kernel function must be defined."
            raise ValueError(errmsg)
        d = np.maximum(d, self.eps)
        if self.logdist:
            d = np.log(d)
        P = None
        for kernel in self.kernels:
            K = kernel(d)
            p = 1 - 1 / (K + 1)
            if P is None:
                P = p
            else:
                P *= p
        if np.isscalar(d):
            P = P.item()
        return 1 - P  # type: ignore

    def sample(
        self, *, sparse: bool = True, seed: int | np.random.Generator | None = None
    ) -> GRGGSample:
        """Sample adjacency matrix from the SRGG model."""
        rng = get_rng(seed)
        points = sphere_surface_sample(self.n, self.sphere.k, seed=rng)
        distances = sphere_distances(points)
        P = self.edgeprob(distances * self.sphere.R)
        np.fill_diagonal(P, 0)
        A = (rng.random(size=P.shape) <= P).astype(int)
        A = np.tril(A) + np.tril(A, -1).T  # make it symmetric
        if sparse:
            A = csr_array(A)
        return GRGGSample(A, points * self.sphere.R)

    def set_kernel(
        self,
        kerntype: type[AbstractGeometricKernel],
        *,
        kbar: float | None = None,
        **kwargs: Any,
    ) -> Self:
        """Set a kernel function for the model.

        Parameters
        ----------
        kerntype : type[AbstractGeometricKernel]
            The kernel class to instantiate.
        kbar : float, optional
            Average degree of the graph. If provided, it will be used to set
            the `mu` parameter of the kernel. When `kbar` is provided,
            setting `mu` will raise an error.
        **kwargs
            Additional parameters for the kernel.
        """
        if kbar is not None and kwargs.get("mu") is not None:
            errmsg = "cannot set both 'kbar' and 'mu'"
            raise ValueError(errmsg)
        kernel = kerntype.from_sphere(self.sphere, **kwargs)
        if kbar is not None:
            rgg = self.copy(kernels=[kernel])
            mu = rgg._estimate_mu_from_kbar(kbar)
            kernel.mu = mu
        self.kernels.append(kernel)
        return self

    def calibrate(self, kbar: float, q: float | np.ndarray | None = None) -> Self:
        """Calibrate the model to have a specific average degree `kbar`.

        Parameters
        ----------
        kbar : float
            Desired average degree of the graph.
        q : float or np.ndarray, optional
            If provided, it will be used to set the relative weights of the kernels.
        """
        q = self._preprocess_q(q)
        mu = self._estimate_mu_from_kbar(kbar, q)
        return self.set_mu(mu, q)

    def set_mu(self, mu: float, q: float | np.ndarray | None = None) -> Self:
        """Set the `mu` parameter for all kernels in the model.

        Parameters
        ----------
        mu : float
            The value to set for the `mu` parameter.
        q : float or np.ndarray, optional
            If provided, it will be used to set the relative weights of the kernels.
        """
        q = self._preprocess_q(q)
        for qi, kernel in zip(q, self.kernels, strict=True):
            kernel.mu = float((mu * qi).item())
        return self

    # Internals ----------------------------------------------------------------------

    def _preprocess_q(self, q: float | np.ndarray | None) -> np.ndarray:
        if q is None:
            q = np.ones(len(self.kernels)) / len(self.kernels)
        if np.isscalar(q):
            if 0 <= q <= 1:  # type: ignore
                q = np.array([q, 1 - q])  # type: ignore
            else:
                errmsg = "if scalar 'q' must be in [0, 1]"
                raise ValueError(errmsg)
        q = np.atleast_1d(q)
        if q.ndim != 1:
            errmsg = "'q' must be a 1D array"
            raise ValueError(errmsg)
        if len(q) != len(self.kernels):
            nk = len(self.kernels)
            errmsg = f"'q' must have the same length as the number of kernels ({nk})"
            raise ValueError(errmsg)
        q = q / q.sum()
        return q

    def _estimate_mu_from_kbar(
        self,
        kbar: float,
        q: float | np.ndarray | None = None,
    ) -> float:
        """Estimate the `mu` parameter from the average degree `kbar`."""
        obj = self.copy()
        q = obj._preprocess_q(q)

        def objective(mu: float) -> float:
            obj.set_mu(mu, q)
            return (obj.kbar - kbar) ** 2

        mu0 = self.sphere.R * np.pi / 2
        solution = minimize(objective, mu0, method="Nelder-Mead")
        return float(solution.x[0])
