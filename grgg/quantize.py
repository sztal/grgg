import warnings

import numpy as np
from sklearn.preprocessing import KBinsDiscretizer


class KMeansQuantizer(KBinsDiscretizer):
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
        if X.ndim == 1:
            X = X[:, np.newaxis]
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

    def transform(self, X):
        if isinstance(X, np.ndarray) and X.ndim == 1:
            X = X[:, np.newaxis]
        return super().transform(X)

    def inverse_transform(self, Xt):
        if isinstance(Xt, np.ndarray) and Xt.ndim == 1:
            Xt = Xt[:, np.newaxis]
        return super().inverse_transform(Xt)
