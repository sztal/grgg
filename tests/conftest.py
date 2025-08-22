import numpy as np
import pytest

from grgg import GRGG, Complementarity, Similarity, Sphere


@pytest.fixture(scope="function")
def rng() -> np.random.Generator:
    return np.random.default_rng(342794234)


@pytest.fixture(scope="session", params=[1, 2, 3])
def dim(request) -> int:
    return request.param


@pytest.fixture(scope="session", params=[25, 100])
def n_nodes(request) -> int:
    return request.param


@pytest.fixture(scope="session", params=[5, 10])
def kbar(request) -> float:
    return request.param


@pytest.fixture(scope="session", params=[None, 0.2])
def weights(request) -> np.ndarray:
    q = request.param
    if q is None:
        return q
    return np.array([q, 1 - q])


@pytest.fixture(scope="session", params=[0, 3, np.inf])
def beta(request) -> float:
    return request.param


@pytest.fixture(scope="session", params=[False, True])
def logspace(request) -> bool:
    return request.param


@pytest.fixture(scope="session", params=[None, 1.0])
def sphere(dim, request) -> Sphere:
    radius = request.param
    if radius is None:
        return dim
    return Sphere(dim, radius)


@pytest.fixture(scope="function")
def model(n_nodes: int, sphere: Sphere) -> GRGG:
    return GRGG(n_nodes, sphere)


@pytest.fixture(scope="session")
def kernel_sim(beta, logspace) -> tuple[float, float, type[Similarity]]:
    return beta, logspace, Similarity


@pytest.fixture(scope="session")
def kernel_comp(beta, logspace) -> tuple[float, float, type[Complementarity]]:
    return beta, logspace, Complementarity


@pytest.fixture(scope="function")
def model_sim(n_nodes, sphere, kernel_sim) -> GRGG:
    beta, logspace, kernel = kernel_sim
    return GRGG(n_nodes, sphere).add_kernel(kernel, beta=beta, logspace=logspace)


@pytest.fixture(scope="function")
def model_comp(n_nodes, sphere, kernel_comp) -> GRGG:
    beta, logspace, kernel = kernel_comp
    return GRGG(n_nodes, sphere).add_kernel(kernel, beta=beta, logspace=logspace)


@pytest.fixture(scope="function")
def model_simcomp(n_nodes, sphere, kernel_sim, kernel_comp) -> GRGG:
    beta_sim, logspace_sim, kernel_sim = kernel_sim
    beta_comp, logspace_comp, kernel_comp = kernel_comp
    return (
        GRGG(n_nodes, sphere)
        .add_kernel(kernel_sim, beta=beta_sim, logspace=logspace_sim)
        .add_kernel(kernel_comp, beta=beta_comp, logspace=logspace_comp)
    )
