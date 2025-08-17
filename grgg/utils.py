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
