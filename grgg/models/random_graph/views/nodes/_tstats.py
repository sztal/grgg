import equinox as eqx
import jax.numpy as jnp

from grgg._typing import Reals
from grgg.statistics import TStatistics

from ._similarity import RandomGraphStructuralSimilarity
from ._tclosure import RandomGraphTClosure
from ._tclust import RandomGraphTClustering


class RandomGraphTStatistics(TStatistics):
    @staticmethod
    @eqx.filter_jit
    def m1_from_motifs(triangles: Reals, twedges: Reals, theads: Reals) -> Reals:
        """Compute the first moment of the statistic from motifs counts."""
        tclust = RandomGraphTClustering.m1_from_motifs(triangles, twedges)
        tclosure = RandomGraphTClosure.m1_from_motifs(triangles, theads)
        similarity = RandomGraphStructuralSimilarity.m1_from_motifs(
            triangles, twedges, theads
        )
        return jnp.stack([tclust, tclosure, similarity])

    def _homogeneous_m1(self) -> Reals:
        return _m1(self)

    def _heterogeneous_m1_exact(self) -> Reals:
        return _m1(self)

    def _heterogeneous_m1_monte_carlo(self) -> Reals:
        return _m1(self)


@eqx.filter_jit
def _m1(stat: RandomGraphTStatistics) -> Reals:
    """Compute the first moment of the statistic."""
    kw1, kw2, kw3 = stat.split_options(3, repeat=1, average=True)
    triangles = stat.nodes.motifs.triangle(**kw1)
    twedges = stat.nodes.motifs.twedge(**kw2)
    theads = stat.nodes.motifs.thead(**kw3)
    return stat.m1_from_motifs(triangles, twedges, theads)
