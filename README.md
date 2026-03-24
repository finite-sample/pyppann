# pyppann

Approximate Nearest Neighbors using Projection Pursuit.

pyppann provides ANN methods that use learned projections to find approximate nearest neighbors. The key innovation is **NeighborPreservingANN**, which learns projections that directly optimize neighbor preservation (recall@k) rather than variance or non-Gaussianity.

## Installation

```bash
pip install pyppann
```

## Quick Start

```python
import numpy as np
from pyppann import NeighborPreservingANN

# Generate sample data
X = np.random.randn(1000, 50)

# Fit the model
model = NeighborPreservingANN(n_components=20, k=10)
model.fit(X, k=10)

# Find k nearest neighbors for queries
queries = np.random.randn(5, 50)
distances, indices = model.kneighbors(queries, n_neighbors=10)
```

## Available Methods

| Class | Description |
|-------|-------------|
| `NeighborPreservingANN` | Learns projections that maximize neighbor preservation. Best recall. |
| `ProjectionPursuitANN` | Uses projection pursuit regression via pyppur. |
| `ANNOYWrapper` | Wrapper around the ANNOY library with sklearn-compatible API. |
| `RandomProjectionANN` | Gaussian random projections + brute kNN. Simple baseline. |
| `PCAAnn` | PCA dimensionality reduction + brute kNN. Simple baseline. |

### Lower-level API

`NeighborPreservingPP` provides a scikit-learn compatible transformer (fit/transform) for the neighbor-preserving projection pursuit algorithm:

```python
from pyppann import NeighborPreservingPP

pp = NeighborPreservingPP(n_components=20, k=10)
X_projected = pp.fit_transform(X)
```

## Evaluation Utilities

pyppann includes utilities for benchmarking ANN methods:

```python
from pyppann import compute_recall, run_comparison, run_full_evaluation

# Compare methods on synthetic data
results = run_comparison(n_components=20, k=10, verbose=True)

# Full evaluation across multiple k and n_components values
all_results = run_full_evaluation(
    n_samples=5000,
    n_features=50,
    k_values=(5, 10, 20),
    n_component_values=(10, 20, 30),
)
```

## License

MIT License
