import math
from copy import copy
from typing import Any

import numpy as np


def get_rng(seed: int | np.random.Generator | None = None) -> np.random.Generator:
    """
    Get a random number generator based on the provided seed.

    Parameters
    ----------
    seed : int or np.random.Generator, optional
        Seed for the random number generator. If None, a default RNG is used.
        If an integer is provided, it initializes a new RNG with that seed.
        If a np.random.Generator instance is provided, it is returned as is.

    Returns
    -------
    np.random.Generator
        A random number generator instance.

    Raises
    ------
    TypeError
        If `seed` is not None, an integer, or a np.random.Generator instance.
    """
    if seed is None:
        return np.random.default_rng()
    if isinstance(seed, int):
        return np.random.default_rng(seed)
    if isinstance(seed, np.random.Generator):
        return seed
    errmsg = (
        "'seed' must be 'None', an 'int', or a 'np.random.Generator' instance, "
        f"not {type(seed)}"
    )
    raise TypeError(errmsg)


def sphere_surface_area(radius: float, k: int) -> float:
    """
    Compute the surface area of a sphere in d-dimensional space.

    Parameters
    ----------
    radius : float
        The radius of the sphere.
    k : int
        Surface dimension of the sphere.

    Returns
    -------
    float
        The surface area of the sphere.

    Examples
    --------
    >>> sphere_surface_area(1, 1)  # doctest: +FLOAT_CMP
    6.283185307179586
    >>> sphere_surface_area(1, 2)  # doctest: +FLOAT_CMP
    12.566370614359172
    """
    k += 1  # use embedding dimension
    return float((2 * math.pi ** (k / 2)) / math.gamma(k / 2) * radius ** (k - 1))


def sphere_volume(radius: float, k: int) -> float:
    """
    Compute the volume of a sphere in d-dimensional space.

    Parameters
    ----------
    radius : float
        The radius of the sphere.
    k : int
        Surface dimension of the space.

    Returns
    -------
    float
        The volume of the sphere.

    Examples
    --------
    >>> sphere_volume(1, 1)  # doctest: +FLOAT_CMP
    3.141592653589793
    >>> sphere_volume(1, 2)  # doctest: +FLOAT_CMP
    4.1887902047863905
    """
    k += 1  # use embedding dimension
    return float((math.pi ** (k / 2)) / math.gamma(k / 2 + 1) * radius**k)


def sphere_radius(S: float, k: int) -> float:
    r"""
    Compute the radius of a sphere given its surface area and dimension.

    Parameters
    ----------
    S : float
        The surface area of the sphere.
    k : int
        Surface dimension of the sphere.

    Returns
    -------
    float
        The radius of the sphere.

    Examples
    --------
    Circe in 2D with area :math:`2\pi`.
    >>> sphere_radius(6.283185307179586, 1)  # doctest: +FLOAT_CMP
    1.0

    Sphere in 3D with area :math:`4\pi`.
    >>> sphere_radius(12.566370614359172, 2)  # doctest: +FLOAT_CMP
    1.0
    """
    k += 1  # use embedding dimension
    return float((S * math.gamma(k / 2) / (2 * math.pi ** (k / 2))) ** (1 / (k - 1)))


def sphere_surface_sample(
    n: int,
    k: int = 1,
    *,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """
    Sample points uniformly from the surface of a unit sphere in k-dimensional space.

    Parameters
    ----------
    n : int
        Number of points to sample.
    k : int, optional
        Surface dimension of the sphere, by default 1 (which is a circle).

    Returns
    -------
    np.ndarray
        An array of shape (n, k) containing the sampled points.

    Examples
    --------
    >>> points = sphere_surface_sample(5, 3, seed=42)
    >>> np.allclose(np.linalg.norm(points, axis=1), 1.0)
    True
    """
    rng = get_rng(seed)
    # Generate random points on the sphere
    X = rng.normal(size=(n, k + 1))
    X /= np.linalg.norm(X, axis=1)[:, None]
    return X


def sphere_distances(X: np.ndarray) -> np.ndarray:
    """
    Compute the pairwise spherical distance between points on the unit sphere.

    Parameters
    ----------
    X : np.ndarray
        An array of shape (n, k) containing points on the sphere.

    Returns
    -------
    np.ndarray
        A square matrix of shape (n, n) containing the pairwise distances.

    Examples
    --------
    >>> X = np.array([[1, 0], [0, 1]])
    >>> sphere_distances(X)  # doctest: +FLOAT_CMP
        array([[0.        , 1.57079],
           [1.57079, 0.        ]])
    """
    return np.arccos(np.clip(np.dot(X, X.T), -1.0, 1.0))


def copy_with_update[T](obj: T, **kwargs: Any) -> T:
    """
    Create a shallow copy of an object and update its attributes.

    Parameters
    ----------
    obj : Any
        The object to copy.
    **kwargs : Any
        Attributes to update in the copied object.

    Returns
    -------
    Any
        A new object with updated attributes.
    """
    new_obj = copy(obj)
    for key, value in kwargs.items():
        if hasattr(new_obj, key):
            setattr(new_obj, key, value)
        else:
            cname = new_obj.__class__.__name__
            errmsg = f"'{cname}' object has no attribute '{key}'"
            raise AttributeError(errmsg)
    return new_obj
