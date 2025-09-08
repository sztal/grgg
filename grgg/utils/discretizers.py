from typing import ClassVar

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.utils._param_validation import (
    Integral,
    Interval,
    Real,
    StrOptions,
    validate_parameter_constraints,
)
from sklearn.utils.validation import check_array, check_is_fitted, validate_data

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
    >>> discretizer = KMeansDiscretizer(std_per_bin=0.2, strategy='independent')
    >>> discretizer.fit(X)
    KMeansDiscretizer(std_per_bin=0.2)
    >>> Y = discretizer.transform(X)
    >>> Y
    array([[0, 0],
           [0, 0],
           [0, 0],
           [1, 1],
           [1, 1]], dtype=int32)
    >>> discretizer.inverse_transform(Y)
    array([[0.28333333, 1.2       ],
           [0.28333333, 1.2       ],
           [0.28333333, 1.2       ],
           [0.95      , 1.9       ],
           [0.95      , 1.9       ]])
    >>> discretizer = KMeansDiscretizer(
    ...     std_per_bin=0.2, strategy='joint', random_state=17
    ... )
    >>> discretizer.fit(X)
    KMeansDiscretizer(random_state=17, std_per_bin=0.2, strategy='joint')
    >>> Y = discretizer.transform(X)
    >>> Y
    array([[0],
           [0],
           [0],
           [2],
           [1]], dtype=int32)
    >>> discretizer.inverse_transform(Y)
    array([[0.28333333, 1.2       ],
           [0.28333333, 1.2       ],
           [0.28333333, 1.2       ],
           [0.9       , 1.8       ],
           [1.        , 2.        ]])
    """

    _parameter_constraints: ClassVar[dict] = {
        "std_per_bin": [Interval(Real, 0, None, closed="neither")],  # type: ignore
        "max_bins": [Interval(Integral, 1, None, closed="left")],  # type: ignore
        "strategy": [StrOptions(frozenset(["independent", "joint"]))],  # type: ignore
    }

    def __init__(
        self,
        std_per_bin: float = 0.05,
        max_bins: int = 512,
        strategy: str = "independent",
        random_state: np.random.Generator | int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        std_per_bin
            Target standard deviation per bin.
        max_bins
            Maximum number of bins (clusters) to use.
        strategy
            Strategy to fit the KMeans models. If 'independent', a separate KMeans
            model is fitted for each feature. If 'joint', a single KMeans model is
            fitted to all features jointly.
        """
        params = {
            "std_per_bin": std_per_bin,
            "max_bins": max_bins,
            "strategy": strategy,
            "random_state": random_state,
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
        kmeans = []
        for feature in X.T:
            std = feature.std()
            n_bins = min(max(1, round(std / self.std_per_bin)), self.max_bins)
            x0, x1 = feature.min(), feature.max()
            init_centers = np.linspace(x0, x1, n_bins).reshape(-1, 1)
            km = KMeans(n_clusters=n_bins, init=init_centers, algorithm="lloyd")
            km.fit(feature[:, None])
            kmeans.append(km)
        return kmeans

    def __fit_joint(self, X: np.ndarray) -> list[KMeans]:
        std = X.var(axis=0).sum() ** 0.5
        n_bins = min(max(1, round(std / self.std_per_bin)), self.max_bins)
        km = KMeans(
            n_clusters=n_bins,
            algorithm="lloyd",
            init="k-means++",
            random_state=self.random_state,
        )
        km.fit(X)
        return [km]

    def transform(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        if self.strategy == "independent":
            Y = [
                km.predict(x[:, None]) for km, x in zip(self.kmeans_, X.T, strict=True)
            ]
        else:
            Y = [self.kmeans_[0].predict(X)]
        return np.column_stack(Y)

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
