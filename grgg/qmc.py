from typing import Any

import numpy as np
from scipy.stats.qmc import QMCEngine, Sobol


class QMCIntTuples:
    """Quasi-Monte Carlo sampler for integer tuples.

    Parameters
    ----------
    d
        Dimension of the tuples.
    engine
        A QMC engine from `scipy.stats.qmc` (e.g., `Halton`, `Sobol`, etc.).
    **kwargs
        Additional keyword arguments passed to the QMC engine constructor
        if passed as a class.

    Examples
    --------
    >>> tuples = QMCIntTuples(3, scramble=False)
    >>> tuples.sample(4, 3)
    array([[0, 0, 0],
           [1, 1, 1],
           [2, 0, 0],
           [0, 2, 2]])
    >>> tuples.sample(4, 3, unique=True)
    array([[1, 0, 2]])
    """

    def __init__(
        self,
        d: int,
        engine: QMCEngine | type[QMCEngine] = Sobol,
        **kwargs: Any,
    ) -> None:
        self.d = int(d)
        if isinstance(engine, type):
            engine = engine(d=d, **kwargs)
        self.engine = engine

    def sample(
        self,
        n: int,
        lbound: int,
        ubound: int | None = None,
        *,
        unique: bool = False,
    ) -> np.ndarray:
        """Sample `n` integer tuples."""
        if ubound is None:
            lbound, ubound = 0, lbound
        sample = self.engine.integers(lbound, u_bounds=ubound, n=n)
        if unique:
            is_unique = np.full(sample.shape[0], True)
            for k in range(sample.shape[1] - 1):
                is_unique &= (sample[:, k, None] != sample[:, k + 1 :]).all(axis=1)
            sample = sample[is_unique]
        return sample
