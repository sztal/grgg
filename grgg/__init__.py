from .__about__ import __version__
from ._options import options
from .manifolds import Sphere
from .model import GRGG, Complementarity, Similarity

__all__ = (
    "GRGG",
    "Complementarity",
    "Similarity",
    "Sphere",
)
