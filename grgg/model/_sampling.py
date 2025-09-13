from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any

import igraph as ig
import jax.numpy as jnp
import numpy as np
from flax import nnx
from pathcensus import PathCensus
from scipy.sparse import csr_array, sparray
from tqdm.auto import tqdm

from grgg._typing import Booleans, BoolVector, Floats, IntVector
from grgg.abc import AbstractGRGG
from grgg.utils import batch_slices, random_state

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


class Sampler(nnx.Module):
    """Sampler for sampling from the GRGG model.

    Attributes
    ----------
    nodes
        Nodes view.
    """

    def __init__(self, nodes: "NodeView") -> None:
        if not isinstance(nodes.module, AbstractGRGG):
            errmsg = "sampling can only be performed on a view of a full model"
            raise TypeError(errmsg)
        super().__init__()
        self.nodes = nodes
        self._sample_diag = nnx.jit(self._sample_diag, static_argnames=("s",))
        self._sample_offdiag = nnx.jit(
            self._sample_offdiag, static_argnames=("s1", "s2")
        )

    def __call__(self, **kwargs: Any) -> Sample:
        return self.sample(**kwargs)

    @property
    def model(self) -> "GRGG":
        """Parent model module."""
        return self.nodes.module

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the model."""
        return self.model.n_nodes

    def sample(self, **kwargs: Any) -> Sample:
        """Sample a graph from the model.

        Parameters
        ----------
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
        >>> import jax.numpy as np
        >>> from grgg import GRGG, Similarity
        >>> from grgg.utils import random_state
        >>> rngs = random_state(42)
        >>> model = GRGG(100, 2) + Similarity(3.0, rngs.normal(100))
        >>> S = model.sample(rngs=rngs, batch_size=50)
        >>> S.A.shape
        (100, 100)
        >>> S.X.shape
        (100, 3)

        Returns
        -------
        Sample
            Sampled graph and associated data.
        """
        points, i, j = (np.asarray(x) for x in self._sample(**kwargs))
        n_nodes = self.n_nodes
        A = csr_array((np.ones_like(i), (i, j)), shape=(n_nodes, n_nodes))
        A += A.T
        return Sample(A, points)

    def _sample(
        self,
        *,
        batch_size: int | None = None,
        rngs: nnx.Rngs | int | None = None,
        progress: bool | Mapping | None = None,
    ) -> tuple[Floats, IntVector, IntVector]:
        nodes = (
            self.model.materialize(copy=False).nodes
            if self.nodes.is_active
            else self.nodes
        )
        batch_size = self.model.get_batch_size(batch_size)
        progress, pkw = self.model.get_progress(progress)
        rngs = random_state(rngs)
        points = self.nodes.sample_points(rngs=rngs)
        n_nodes = nodes.n_nodes
        bslices = [
            (bs1, bs2)
            for bs1, bs2 in batch_slices(n_nodes, batch_size, repeat=2)
            if bs1 <= bs2  # type: ignore
        ]
        Ai = []
        Aj = []
        for bs1, bs2 in tqdm(bslices, disable=not progress, **pkw):
            if bs1 > bs2:  # type: ignore
                continue
            if bs1 == bs2:
                i, j, M = self._sample_diag(points, bs1, rngs)
                i = i[M]
                j = j[M]
            else:
                M = self._sample_offdiag(points, bs1, bs2, rngs)
                i, j = jnp.where(M)
            i += bs1.start
            j += bs2.start
            Ai.append(i)
            Aj.append(j)
        Ai = np.concatenate(Ai)
        Aj = np.concatenate(Aj)
        return points, Ai, Aj

    def _sample_diag(
        self,
        points: Floats,
        s: slice,
        rngs: nnx.Rngs,
    ) -> tuple[IntVector, IntVector, BoolVector]:
        X = points[s]
        g = self.model.manifold.distances(X, condensed=True)
        i, j = jnp.triu_indices(len(X), k=1)
        P = self.model.pairs[i, j].probs(g)
        M = rngs.uniform(P.shape) < P
        return i, j, M

    def _sample_offdiag(
        self,
        points: Floats,
        s1: slice,
        s2: slice,
        rngs: nnx.Rngs,
    ) -> Booleans:
        X1 = points[s1]
        X2 = points[s2]
        g = self.model.manifold.distances(X1, X2)
        P = self.model.pairs[s1, s2].probs(g)
        M = rngs.uniform(P.shape) < P
        return M
