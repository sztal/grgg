import equinox as eqx
import jax.numpy as jnp

from grgg._typing import Reals
from grgg.statistics import QStatistics

from ._complementarity import RandomGraphStructuralComplementarity
from ._qclosure import RandomGraphQClosure
from ._qclust import RandomGraphQClustering


class RandomGraphQStatistics(QStatistics):
    @staticmethod
    @eqx.filter_jit
    def m1_from_motifs(quadrangles: Reals, qwedges: Reals, qheads: Reals) -> Reals:
        """Compute the first moment of the statistic from motifs counts."""
        qclust = RandomGraphQClustering.m1_from_motifs(quadrangles, qwedges)
        qclosure = RandomGraphQClosure.m1_from_motifs(quadrangles, qheads)
        complementarity = RandomGraphStructuralComplementarity.m1_from_motifs(
            quadrangles, qwedges, qheads
        )
        return jnp.stack([qclust, qclosure, complementarity])

    def _homogeneous_m1(self) -> Reals:
        return _m1(self)

    def _heterogeneous_m1_exact(self) -> Reals:
        return _m1(self)

    def _heterogeneous_m1_monte_carlo(self) -> Reals:
        return _m1(self)


@eqx.filter_jit
def _m1(stat: RandomGraphQStatistics) -> Reals:
    """Compute the first moment of the statistic."""
    kw1, kw2, kw3 = stat.split_options(3, repeat=1, average=True)
    quadrangles = stat.nodes.motifs.quadrangle(**kw1)
    qwedges = stat.nodes.motifs.qwedge(**kw2)
    qheads = stat.nodes.motifs.qhead(**kw3)
    return stat.m1_from_motifs(quadrangles, qwedges, qheads)
