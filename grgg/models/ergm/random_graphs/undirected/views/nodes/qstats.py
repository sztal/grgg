from typing import Any

import jax.numpy as jnp

from grgg._typing import Reals
from grgg.statistics import QStatistics

from .complementarity import UndirectedRandomGraphStructuralComplementarity
from .qclosure import UndirectedRandomGraphQClosure
from .qclust import UndirectedRandomGraphQClustering


class UndirectedRandomGraphQStatistics(QStatistics):
    """Quadrangle-based statistic for undirected random graphs."""

    @staticmethod
    def m1_from_motifs(quadrangles: Reals, fwedges: Reals, fheads: Reals) -> Reals:
        """Compute the first moment of the statistic from motifs counts."""
        qclust = UndirectedRandomGraphQClustering.m1_from_motifs(
            quadrangles, fwedges, fheads
        )
        qclosure = UndirectedRandomGraphQClosure.m1_from_motifs(
            quadrangles, fwedges, fheads
        )
        complementarity = UndirectedRandomGraphStructuralComplementarity.m1_from_motifs(
            quadrangles, fwedges, fheads
        )
        return jnp.stack([qclust, qclosure, complementarity], axis=-1)

    def _m1(self, **kwargs: Any) -> Reals:
        """Compute the first moment of the statistic."""
        kw1, kw2, kw3 = self.split_compute_kwargs(3, same_seed=True, **kwargs)
        quadrangles = self.nodes.motifs.quadrangle(**kw1)
        fwedges = self.nodes.motifs.fwedge(**kw2)
        fheads = self.nodes.motifs.fhead(**kw3)
        return self.m1_from_motifs(quadrangles, fwedges, fheads)

    def _homogeneous_m1(self, **kwargs: Any) -> Reals:
        return self._m1(**kwargs)

    def _heterogeneous_m1(self, **kwargs: Any) -> Reals:
        return self._m1(**kwargs)
