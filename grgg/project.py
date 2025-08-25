from pathlib import Path
from types import SimpleNamespace

import dvc.api
import igraph
from dotenv import load_dotenv
from omegaconf import OmegaConf

load_dotenv()  # take environment variables from .env, if defined


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


paths = make_paths()
config = OmegaConf.create(dvc.api.params_show())
OmegaConf.resolve(config)

# Set igraph config ------------------------------------------------------------------

igraph.config["plotting.backend"] = "matplotlib"
