import warnings
from typing import ClassVar

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._param_validation import (
    Integral,
    Interval,
    StrOptions,
    validate_parameter_constraints,
)
from sklearn.utils.validation import check_array, check_is_fitted, validate_data

from .functions import make_grid

__all__ = ("KMeansDiscretizer",)


class KMeansDiscretizer(BaseEstimator, TransformerMixin):
    """K-means based discretization of continuous features.

    Attributes
    ----------
    kmeans_
        Array fitted KMeans models, one per feature if `strategy` is 'independent',
        otherwise an array of length one with one KMeans model fitted to all features.
    std_
        Within-cluster standard deviation averaged over features
        for each KMeans model.
    n_bins_
        Number of bins (clusters) for each KMeans model.
    n_features_in_
        Number of features seen during `fit`.
    names_features_in_
        Names of features seen during `fit` (if available).

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[0.1, 1.0],
    ...               [0.4, 1.2],
    ...               [0.35, 1.4],
    ...               [0.9, 1.8],
    ...               [1.0, 2.0]])
    >>> discretizer = KMeansDiscretizer(n_bins=3, strategy="independent")
    >>> discretizer.fit(X)
    KMeansDiscretizer(n_bins=3, strategy='independent')
    >>> Y = discretizer.transform(X)
    >>> Y
    array([[0, 0],
           [1, 0],
           [1, 1],
           [2, 2],
           [2, 2]], dtype=int32)
    >>> discretizer.inverse_transform(Y)
    array([[0.1  , 1.1  ],
           [0.375, 1.1  ],
           [0.375, 1.4  ],
           [0.95 , 1.9  ],
           [0.95 , 1.9  ]])
    >>> discretizer = KMeansDiscretizer(n_bins=3, strategy="joint")
    >>> discretizer.fit(X)
    KMeansDiscretizer(n_bins=3)
    >>> Y = discretizer.transform(X)
    >>> Y
    array([[0],
           [2],
           [1],
           [3],
           [3]], dtype=int32)

    >>> discretizer.inverse_transform(Y)
    array([[0.1 , 1.  ],
           [0.4 , 1.2 ],
           [0.35, 1.4 ],
           [0.95, 1.9 ],
           [0.95, 1.9 ]])
    """

    _parameter_constraints: ClassVar[dict] = {
        "n_bins": [Interval(Integral, 1, None, closed="left")],  # type: ignore
        "strategy": [StrOptions(frozenset(["independent", "joint"]))],  # type: ignore
    }

    def __init__(
        self,
        n_bins: int = 256,
        strategy: str = "joint",
    ) -> None:
        """
        Parameters
        ----------
        n_bins
            Target number of bins (clusters). If `strategy` is 'independent',
            the actual number of bins may be signficantly larger than `n_bins`
            depending on the number of features and the data distribution.
        strategy
            Strategy to fit the KMeans models. If 'independent', a separate KMeans
            model is fitted for each feature. If 'joint', a single KMeans model is
            fitted to all features jointly.
        """
        params = {
            "n_bins": n_bins,
            "strategy": strategy,
        }
        validate_parameter_constraints(
            self._parameter_constraints, params, self.__class__.__name__
        )
        self.__dict__.update(params)

    def set_params(self, **params):
        validate_parameter_constraints(
            self._parameter_constraints, params, self.__class__.__name__
        )
        super().set_params(**params)
        return self

    def fit(self, X, y=None):
        X = validate_data(self, X, y, reset=True)
        fit = (
            self.__fit_independent
            if self.strategy == "independent"
            else self.__fit_joint
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.kmeans_ = np.array(fit(X), dtype=object)
        self.std_ = np.array(
            [
                np.sqrt(km.inertia_ / len(km.labels_) / km.cluster_centers_.shape[1])
                for km in self.kmeans_
            ]
        )
        self.n_bins_ = np.array([len(km.cluster_centers_) for km in self.kmeans_])
        self.n_features_in_ = X.shape[1]
        if all(
            getattr(km, "features_names_in_", None) is not None for km in self.kmeans_
        ):
            self.feature_names_in_ = np.array(
                sum(list(km.features_names_in_) for km in self.kmeans_)
            ).flatten()
        return self

    def __fit_independent(self, X: np.ndarray) -> list[KMeans]:
        def fit(n_bins: int, feature: np.ndarray) -> KMeans:
            x0, x1 = feature.min(), feature.max()
            init_centers = np.linspace(x0, x1, n_bins).reshape(-1, 1)
            km = KMeans(n_clusters=n_bins, init=init_centers, algorithm="lloyd")
            km.fit(feature[:, None])
            return km

        n_bins = self.__get_n_bins(X)
        kmeans = []
        for feature in X.T:
            km = fit(n_bins, feature)
            n_clusters = len(np.unique(km.labels_))
            if max(km.labels_) >= n_clusters:
                km = fit(n_clusters, feature)
            kmeans.append(km)
        return kmeans

    def __fit_joint(self, X: np.ndarray) -> list[KMeans]:
        def fit(n_bins: int, X: np.ndarray) -> KMeans:
            n_bins = min(n_bins, len(X))
            ranges = [(x.min(), x.max()) for x in X.T]
            init_centers = make_grid(n_bins, self.n_features_in_, ranges)
            if len(init_centers) > len(X):
                n_bins = int(np.ceil(len(X) ** (1 / self.n_features_in_)))
                init_centers = make_grid(n_bins, self.n_features_in_, ranges)
            n_bins = len(init_centers)
            km = KMeans(n_clusters=n_bins, init=init_centers, algorithm="lloyd")
            km.fit(X)
            return km

        n_bins = self.__get_n_bins(X)
        km = fit(n_bins, X)
        n_clusters = len(np.unique(km.labels_))
        if max(km.labels_) >= n_clusters:
            km = fit(n_clusters, X)
        return [km]

    def __get_n_bins(self, X: np.ndarray) -> int:
        n_bins = self.n_bins
        return min(n_bins, len(X))

    def transform(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        if self.strategy == "independent":
            Y = [
                km.predict(x[:, None]) for km, x in zip(self.kmeans_, X.T, strict=True)
            ]
        else:
            Y = [self.kmeans_[0].predict(X)]
        codes = np.column_stack(Y)
        return codes

    def inverse_transform(self, X):
        check_is_fitted(self)
        if self.strategy == "independent":
            X = validate_data(self, X, reset=False)
        else:
            if X.ndim != 2 and X.shape[1] != 1:
                errmsg = (
                    "X should be a 2D array with a single column when using "
                    "strategy='joint'."
                )
                raise ValueError(errmsg)
            X = check_array(X)
        if self.strategy == "independent":
            Y = [
                km.cluster_centers_[x] for km, x in zip(self.kmeans_, X.T, strict=True)
            ]
        else:
            Y = [self.kmeans_[0].cluster_centers_[X.flatten()]]
        return np.column_stack(Y)
