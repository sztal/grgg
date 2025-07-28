from pathlib import Path
from types import SimpleNamespace

import igraph

from .__about__ import __version__
from ._options import options
from .kernels import Complementarity, Similarity
from .manifolds import Sphere
from .model import GRGG

__all__ = (
    "GRGG",
    "Sphere",
    "Complementarity",
    "Similarity",
)


def make_paths(root: str | Path | None = None) -> SimpleNamespace:
    root = Path(__file__).parent.parent if root is None else Path(root).absolute()
    paths = SimpleNamespace(
        root=root,
        figures=root / "figures",
        data=root / "data",
        raw=root / "data" / "raw",
        proc=root / "data" / "proc",
    )
    paths.figures.mkdir(exist_ok=True, parents=True)
    paths.raw.mkdir(exist_ok=True, parents=True)
    paths.proc.mkdir(exist_ok=True, parents=True)
    return paths


# Set igraph config ------------------------------------------------------------------

igraph.config["plotting.backend"] = "matplotlib"
