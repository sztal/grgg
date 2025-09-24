from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property, partial
from typing import TYPE_CHECKING, Any

import equinox as eqx
import igraph as ig
import jax
import jax.numpy as jnp
import numpy as np
from pathcensus import PathCensus
from scipy.sparse import csr_array, sparray
from tqdm.auto import tqdm

from grgg._typing import Booleans, BoolVector, Floats, IntVector
from grgg.abc import AbstractGRGG
from grgg.random import RandomGenerator
from grgg.utils import batch_slices

if TYPE_CHECKING:
    from ._views import NodeView
    from .grgg import GRGG


@dataclass(frozen=True)
class Sample:
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
    X: jnp.ndarray

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


class Sampler(eqx.Module):
    """Sampler for sampling from the GRGG model.

    Attributes
    ----------
    nodes
        Nodes view.
    """

    nodes: "NodeView"

    def __init__(self, nodes: "NodeView") -> None:
        if not isinstance(nodes.model, AbstractGRGG):
            errmsg = (
                "sampling can only be performed for GRGG models, not layer modules; "
                "extract a layer-specific submodule to sample based on a single layer"
            )
            raise TypeError(errmsg)
        super().__init__()
        self.nodes = nodes

    def __call__(self, **kwargs: Any) -> Sample:
        return self.sample(**kwargs)

    @property
    def model(self) -> "GRGG":
        """Parent model module."""
        return self.nodes.model

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the model."""
        return self.nodes.n_nodes

    def sample(self, points: jnp.ndarray | None = None, **kwargs: Any) -> Sample:
        """Sample a graph from the model.

        Parameters
        ----------
        points
            Optional pre-sampled points on the manifold.
            Must be of shape `(n_nodes, dim+1)`.
            If `None`, points are sampled from the model.
        batch_size
            Batch size for processing node pairs.
            If less than or equal to 0, process all node pairs at once.
        rngs
            Random state for reproducibility, can be an integer seed,
            a `nnx.Rngs` object, or `None` for random initialization.
        progress
            Whether to display a progress bar. Can be a boolean or a
            dictionary of keyword arguments passed to `tqdm.tqdm`.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from grgg import GRGG, Similarity
        >>> from grgg.random import RandomGenerator
        >>> rng = RandomGenerator.from_seed(42)
        >>> model = GRGG(100, 2) + Similarity(rng.normal(100), 3.0)
        >>> S = model.sample(rng=rng, batch_size=50)
        >>> S.A.shape
        (100, 100)
        >>> S.X.shape
        (100, 3)

        It is also possible to sample from a node view of the model.
        >>> S = model.nodes[:10].sample(rng=rng)
        >>> S.A.shape
        (10, 10)
        >>> S.X.shape
        (10, 3)

        Sampling from a quantized model is allowed as well,
        even though the meaning of such an operation is less clear.
        >>> S = model.quantize(n_codes=16).sample(rng=rng)
        >>> S.A.shape
        (16, 16)
        >>> S.X.shape
        (16, 3)

        Returns
        -------
        Sample
            Sampled graph and associated data.
        """
        points, i, j = self._sample(**kwargs)
        i, j = jnp.asarray(i), jnp.asarray(j)
        n_nodes = len(points)
        A = csr_array((np.ones_like(i), (i, j)), shape=(n_nodes, n_nodes))
        A += A.T
        return Sample(A, points)

    def _sample(
        self,
        points: jnp.ndarray | None = None,
        *,
        batch_size: int | None = None,
        rng: RandomGenerator | int | None = None,
        progress: bool | Mapping | None = None,
    ) -> tuple[Floats, IntVector, IntVector]:
        if self.nodes.is_active:
            nodes = self.nodes.materialize(copy=False).nodes
            self = self.__class__(nodes)
        else:
            nodes = self.nodes
        batch_size = self.model._get_batch_size(batch_size)
        progress, pkw = self.model._get_progress(progress)
        rng = RandomGenerator.from_seed(rng)
        if points is None:
            points = self.nodes.sample_points(rng=rng)
        else:
            expected_shape = (nodes.n_nodes, self.model.manifold.embedding_dim)
            if points.shape != expected_shape:
                errmsg = (
                    f"'points' must be of shape {expected_shape}, got {points.shape}"
                )
                raise ValueError(errmsg)
        n_nodes = nodes.n_nodes
        bslices = [
            (bs1, bs2)
            for bs1, bs2 in batch_slices(n_nodes, batch_size, repeat=2)
            if bs1 <= bs2  # type: ignore
        ]
        Ai = []
        Aj = []
        for bs1, bs2 in tqdm(bslices, disable=not progress, **pkw):
            if bs1 == bs2:
                i, j, M = _sample_diag(self, points, bs1, rng)
                i = i[M]
                j = j[M]
            else:
                M = _sample_offdiag(self, points, bs1, bs2, rng)
                i, j = jnp.where(M)
            i += bs1.start
            j += bs2.start
            Ai.append(i)
            Aj.append(j)
        Ai = jnp.concatenate(Ai)
        Aj = jnp.concatenate(Aj)
        return points, Ai, Aj


@partial(jax.jit, static_argnames=("s",))
def _sample_diag(
    sampler,
    points: Floats,
    s: slice,
    rng: RandomGenerator,
) -> tuple[IntVector, IntVector, BoolVector]:
    X = points[s]
    g = sampler.model.manifold.distances(X, condensed=True)
    i, j = jnp.triu_indices(len(X), k=1)
    P = sampler.model.pairs[i, j].probs(g)
    M = rng.uniform(P.shape) < P
    return i, j, M


@partial(jax.jit, static_argnames=("s1", "s2"))
def _sample_offdiag(
    sampler,
    points: Floats,
    s1: slice,
    s2: slice,
    rng: RandomGenerator,
) -> Booleans:
    X1 = points[s1]
    X2 = points[s2]
    g = sampler.model.manifold.distances(X1, X2)
    P = sampler.model.pairs[s1, s2].probs(g)
    M = rng.uniform(P.shape) < P
    return M
