---
applyTo: "grgg/**/*.py,tests/**/*.py,scripts/**/*.py,analyses/**/*.py"
---

# GRGG: Generalized Random Geometric Graphs - AI Coding Instructions

## Project Overview
GRGG implements Generalized Random Geometric Graphs - mathematical models for networks where nodes are embedded in geometric spaces and edge probabilities depend on spatial relationships. The framework supports both similarity (proximity-based) and complementarity (dissimilarity-based) connections.

## Conventions

### Python Code Style
- **Docstrings**: Use triple quotes `"""` with NumPy-style format including Parameters, Returns, Examples sections
- **Strings**: Always use double quotes `"string"` for consistency (enforced by Ruff Q rule)
- **Type hints**: Mandatory for all function arguments and return types, leveraging `typing` and `jaxtyping`
- **Naming**: snake_case for variables/functions, PascalCase for classes, UPPER_CASE for constants
- **Error messages**: Use descriptive error messages stored in variables like `errmsg = "descriptive message"`. Exceptions should be specific and raised from named variables,
not inline strings.

### Testing Patterns
- **pytest**: Primary testing framework with parametrized tests using `@pytest.mark.parametrize`
- **Doctests**: Enabled via `--doctest-modules` - include executable examples in docstrings


## Architecture & Key Components

### JAX-Based Computing Stack
- **JAX ecosystem**: Uses `jax.numpy` instead of numpy, `flax.nnx` for computation modules
- **Type system**: Leverages `jaxtyping` for array shape annotations (`Vector`, `Matrix`, `Scalar`)
- **Performance**: Critical functions are JIT-compiled with `@nnx.jit` decorators
- **Arrays**: All arrays are JAX arrays, not NumPy - use `jax.numpy as np` consistently

### Core Architecture Pattern
```python
# Main entry point - compositional model building
# NOTE: THIS SECTION NEEDS TO BE UPDATED LATER

# Models contain:
# - manifold: geometric space (currently only Sphere)
# - layers: kernel functions (Similarity/Complementarity)
# - functions: coupling and probability transformations
```

### Module Hierarchy
- `grgg.model.model.GRGG`: Main model class implementing `Sequence[AbstractLayer]`
- `grgg.model.layers.{Similarity,Complementarity}`: Layer function implementations
- `grgg.model.manifolds.Sphere`: Geometric embedding space
- `grgg.model.functions.{CouplingFunction,ProbabilityFunction}`: Distance-to-probability transforms

### Key Design Patterns

**Flax NNX Modules**: All core components inherit from `nnx.Module` for JAX compatibility

```python
class AbstractLayer(AbstractModelModule):  # <- nnx.Module
    beta: Beta  # <- nnx.Param
    mu: Mu      # <- nnx.Param
```

**Lazy Evaluation**: `LazyBroadcast` and `LazyOuter` classes defer expensive pairwise computations

```python
# Used for memory-efficient pairwise operations on large node sets
```

**Single Dispatch**: Methods use `@singledispatchmethod` for type-specific behavior
```python
@singledispatchmethod
def _make_manifold(self, manifold, *args, **kwargs):
    # Handle different manifold input types
```

## Development Workflow

### Environment Setup
```bash
conda env create -f environment.yaml && conda activate grgg
make init  # Sets up git, dvc, pre-commit hooks
```

### Testing & Quality
```bash
make test      # pytest with JAX-specific test patterns
make coverage  # Test coverage analysis
make lint      # Ruff linting
```

### Data Pipeline (DVC)
```bash
NOTE: THIS SECTION NEEDS TO BE UPDATED LATER
```

### Import Patterns
```python
import jax.numpy as jnp  # NOT numpy as np
from flax import nnx
from jaxtyping import Array, Float
```

### Type Hints
```python
# Use project-specific types from _typing.py
Vector = Float[Array, "#values"]
Matrix = Float[Array, "#rows #cols"]
Scalar = Float[Array, ""]
```

### Parameter Classes
```python
# Model parameters are nnx.Param subclasses with constraints
class Beta(AbstractModelParameter):  # Inverse temperature
class Mu(AbstractModelParameter):    # Chemical potential
```

### JAX Compilation
- Use `@nnx.jit` for performance-critical functions
- Pairwise operations use `jax.vmap` for vectorization
- Random number generation uses `nnx.Rngs` and JAX PRNG keys

### Documentation Standards
- **NumPy docstring format**: Include Parameters, Returns, Examples sections
- **Executable examples**: All docstring examples should be runnable (tested via doctests)
- **Mathematical notation**: Use raw strings `r"""` when including LaTeX math expressions
- **Type annotations**: Document array shapes using jaxtyping syntax in docstrings

### Code Quality Tools
- **Ruff**: Comprehensive linting with strict rule set (pycodestyle, pyflakes, isort, etc.)
- **MyPy**: Type checking with numpy plugin enabled
- **Coverage**: Branch coverage tracking with exclusions for debug/abstract code

## Integration Points

### External Dependencies
- **igraph**: Graph objects in samples (`sample.G`)
- **scipy.sparse**: Adjacency matrices (`csr_array`)
- **DVC**: Computational pipeline management
- **pathcensus**: Graph motif analysis

### Memory Management
- Uses `ArrayQuantizer` for large-scale simulations
- `batch_size` parameters for chunked processing
- Lazy evaluation for pairwise computations

## Testing Patterns
- Parametrized tests with multiple random seeds
- JAX-specific numerical tolerance patterns
- Property-based testing for mathematical invariants

## Key Files to Reference
- `grgg/model/model.py`: Main GRGG class and API
- `grgg/model/_typing.py`: JAX array type definitions
- `grgg/model/_utils.py`: JAX utility functions (pairwise, random_state)
- `grgg/model/layers.py`: Kernel function implementations
- `tests/test_quantize.py`: Example testing patterns
