import warnings

import numpy as np
from sklearn.preprocessing import KBinsDiscretizer


class AdaptiveBinsDiscretizer(KBinsDiscretizer):
    """Adaptive binning of continuous features.

    The number of bins is adapted based on the average standard deviation of the
    features to ensure that each bin has a reasonable width.
    It extends and modifies :class:`sklearn.preprocessing.KBinsDiscretizer`
    through the following parameters.

    Parameters
    ----------
    n_bins
        Maximum number of bins to produce.
        The actual number of bins may be smaller if the variation in the data
        is small enough to be captured with fewer bins.
    average_std_per_bin
        Target average standard deviation per bin.
        The actual number of bins is determined by dividing the average standard
        deviation of the features by this value, and rounding to the nearest
        integer between 2 and `n_bins`. Moreover, constant features will be grouped
        in one bin.
    """

    def __init__(
        self,
        n_bins=100,
        *,
        encode="ordinal",
        strategy="kmeans",
        quantile_method="warn",
        dtype=None,
        subsample=200000,
        random_state=None,
        average_std_per_bin=0.1,
    ) -> None:
        super().__init__(
            n_bins=n_bins,
            encode=encode,
            strategy=strategy,
            quantile_method=quantile_method,
            dtype=dtype,
            subsample=subsample,
            random_state=random_state,
        )
        self.average_std_per_bin = average_std_per_bin

    def fit(self, X, y=None, sample_weight=None):
        if self.average_std_per_bin is not None and self.average_std_per_bin > 0:
            stdev = X.std(axis=0).mean()
            self.n_bins = min(
                max(2, round(stdev / self.average_std_per_bin)), self.n_bins
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            super().fit(X, y, sample_weight)
        for i, edge in enumerate(self.bin_edges_):
            if np.isinf(edge).all():
                edge[:] = [X[:, i].min(), X[:, i].max()]
        return self
