from typing import Any

import jax.numpy as jnp

from grgg._typing import Reals
from grgg.statistics import TStatistics

from .similarity import UndirectedRandomGraphStructuralSimilarity
from .tclosure import UndirectedRandomGraphTClosure
from .tclust import UndirectedRandomGraphTClustering


class UndirectedRandomGraphTStatistics(TStatistics):
    """Triangle-based statistic for undirected random graphs."""

    @staticmethod
    def m1_from_motifs(triangles: Reals, twedges: Reals, theads: Reals) -> Reals:
        """Compute the first moment of the statistic from motifs counts."""
        tclust = UndirectedRandomGraphTClustering.m1_from_motifs(
            triangles, twedges, theads
        )
        tclosure = UndirectedRandomGraphTClosure.m1_from_motifs(
            triangles, twedges, theads
        )
        similarity = UndirectedRandomGraphStructuralSimilarity.m1_from_motifs(
            triangles, twedges, theads
        )
        return jnp.stack([tclust, tclosure, similarity], axis=-1)

    def _m1(self, **kwargs: Any) -> Reals:
        """Compute the first moment of the statistic."""
        kw1, kw2, kw3 = self.split_compute_kwargs(3, same_seed=True, **kwargs)
        triangles = self.nodes.motifs.triangle(**kw1)
        twedges = self.nodes.motifs.twedge(**kw2)
        theads = self.nodes.motifs.thead(**kw3)
        return self.m1_from_motifs(triangles, twedges, theads)

    def _homogeneous_m1(self, **kwargs: Any) -> Reals:
        return self._m1(**kwargs)

    def _heterogeneous_m1(self, **kwargs: Any) -> Reals:
        return self._m1(**kwargs)
