from typing import Any

import equinox as eqx
import jax.numpy as jnp

from grgg._typing import Reals
from grgg.statistics import QStatistics

from ._complementarity import UndirectedRandomGraphStructuralComplementarity
from ._qclosure import UndirectedRandomGraphQClosure
from ._qclust import UndirectedRandomGraphQClustering


class UndirectedRandomGraphQStatistics(QStatistics):
    """Quadrangle-based statistic for undirected random graphs."""

    @staticmethod
    @eqx.filter_jit
    def m1_from_motifs(quadrangles: Reals, qwedges: Reals, qheads: Reals) -> Reals:
        """Compute the first moment of the statistic from motifs counts."""
        qclust = UndirectedRandomGraphQClustering.m1_from_motifs(quadrangles, qwedges)
        qclosure = UndirectedRandomGraphQClosure.m1_from_motifs(quadrangles, qheads)
        complementarity = UndirectedRandomGraphStructuralComplementarity.m1_from_motifs(
            quadrangles, qwedges, qheads
        )
        return jnp.stack([qclust, qclosure, complementarity])

    def _homogeneous_m1(self, **kwargs: Any) -> Reals:
        return _m1(self, **kwargs)

    def _heterogeneous_m1(self, **kwargs: Any) -> Reals:
        return _m1(self, **kwargs)


@eqx.filter_jit
def _m1(stat: UndirectedRandomGraphQStatistics, **kwargs: Any) -> Reals:
    """Compute the first moment of the statistic."""
    kw1, kw2, kw3 = stat.split_compute_kwargs(3, same_seed=True, **kwargs)
    quadrangles = stat.nodes.motifs.quadrangle(**kw1)
    qwedges = stat.nodes.motifs.qwedge(**kw2)
    qheads = stat.nodes.motifs.qhead(**kw3)
    return stat.m1_from_motifs(quadrangles, qwedges, qheads)
