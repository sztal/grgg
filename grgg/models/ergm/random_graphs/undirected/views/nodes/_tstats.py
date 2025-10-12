from typing import Any

import equinox as eqx
import jax.numpy as jnp

from grgg._typing import Reals
from grgg.statistics import TStatistics

from ._similarity import UndirectedRandomGraphStructuralSimilarity
from ._tclosure import UndirectedRandomGraphTClosure
from ._tclust import UndirectedRandomGraphTClustering


class UndirectedRandomGraphTStatistics(TStatistics):
    """Triangle-based statistic for undirected random graphs."""

    @staticmethod
    @eqx.filter_jit
    def m1_from_motifs(triangles: Reals, twedges: Reals, theads: Reals) -> Reals:
        """Compute the first moment of the statistic from motifs counts."""
        tclust = UndirectedRandomGraphTClustering.m1_from_motifs(triangles, twedges)
        tclosure = UndirectedRandomGraphTClosure.m1_from_motifs(triangles, theads)
        similarity = UndirectedRandomGraphStructuralSimilarity.m1_from_motifs(
            triangles, twedges, theads
        )
        return jnp.stack([tclust, tclosure, similarity])

    def _homogeneous_m1(self, **kwargs: Any) -> Reals:
        return self._m1(**kwargs)

    def _heterogeneous_m1(self, **kwargs: Any) -> Reals:
        return self._m1(**kwargs)


@eqx.filter_jit
def _m1(stat: UndirectedRandomGraphTStatistics, **kwargs: Any) -> Reals:
    """Compute the first moment of the statistic."""
    kw1, kw2, kw3 = stat.split_compute_kwargs(3, same_seed=True, **kwargs)
    triangles = stat.nodes.motifs.triangle(**kw1)
    twedges = stat.nodes.motifs.twedge(**kw2)
    theads = stat.nodes.motifs.thead(**kw3)
    return stat.m1_from_motifs(triangles, twedges, theads)
