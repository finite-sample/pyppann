"""Evaluation utilities for comparing ANN methods.

Provides functions to benchmark different ANN approaches on synthetic
and real datasets, measuring recall@k, fit time, and query time.
"""

from __future__ import annotations

import time
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import make_classification
from sklearn.neighbors import NearestNeighbors

from pyppann.baselines import PCAAnn, RandomProjectionANN
from pyppann.neighbor_preserving_pp import NeighborPreservingANN


class ANNModel(Protocol):
    """Protocol for ANN models."""

    def fit(self, X: NDArray[np.floating], k: int) -> Any: ...

    def kneighbors(
        self,
        X: NDArray[np.floating],
        n_neighbors: int | None = None,
        return_distance: bool = True,
    ) -> tuple[NDArray[np.floating], NDArray[np.intp]] | NDArray[np.intp]: ...


def compute_recall(
    true_neighbors: NDArray[np.intp],
    pred_neighbors: NDArray[np.intp],
) -> float:
    """Compute recall@k between true and predicted neighbors.

    Parameters
    ----------
    true_neighbors : ndarray of shape (n_samples, k)
        True k-nearest neighbor indices.
    pred_neighbors : ndarray of shape (n_samples, k)
        Predicted k-nearest neighbor indices.

    Returns
    -------
    recall : float
        Mean recall@k across all samples.
    """
    n_samples = true_neighbors.shape[0]
    k = true_neighbors.shape[1]

    total_recall = 0.0
    for i in range(n_samples):
        true_set = set(true_neighbors[i])
        pred_set = set(pred_neighbors[i])
        total_recall += len(true_set & pred_set) / k

    return total_recall / n_samples


def create_synthetic_data(
    n_samples: int = 5000,
    n_features: int = 50,
    n_informative: int = 30,
    random_state: int = 42,
) -> NDArray[np.floating]:
    """Create synthetic high-dimensional data for ANN evaluation.

    Parameters
    ----------
    n_samples : int, default=5000
        Number of samples.
    n_features : int, default=50
        Total number of features.
    n_informative : int, default=30
        Number of informative features.
    random_state : int, default=42
        Random seed.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Synthetic data.
    """
    X, _ = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_features - n_informative,
        n_clusters_per_class=3,
        random_state=random_state,
    )
    return X.astype(np.float64)


def evaluate_method(
    model: ANNModel,
    X_train: NDArray[np.floating],
    X_query: NDArray[np.floating],
    true_neighbors: NDArray[np.intp],
    k: int,
    name: str,
) -> dict[str, Any]:
    """Evaluate a single ANN method.

    Parameters
    ----------
    model : ANNModel
        The ANN model to evaluate.
    X_train : ndarray
        Training data.
    X_query : ndarray
        Query data.
    true_neighbors : ndarray
        True k-nearest neighbors for queries.
    k : int
        Number of neighbors.
    name : str
        Name of the method for reporting.

    Returns
    -------
    results : dict
        Dictionary with recall, fit_time, query_time.
    """
    fit_start = time.perf_counter()
    model.fit(X_train, k=k)
    fit_time = time.perf_counter() - fit_start

    query_start = time.perf_counter()
    result = model.kneighbors(X_query, n_neighbors=k, return_distance=False)
    query_time = time.perf_counter() - query_start
    pred_neighbors = np.asarray(result)

    recall = compute_recall(true_neighbors, pred_neighbors)

    return {
        "method": name,
        "recall@k": recall,
        "fit_time_s": fit_time,
        "query_time_s": query_time,
    }


def run_comparison(
    X: NDArray[np.floating] | None = None,
    n_components: int = 20,
    k: int = 10,
    n_queries: int = 500,
    random_state: int = 42,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Run comparison of all ANN methods.

    Parameters
    ----------
    X : ndarray | None
        Data to use. If None, creates synthetic data.
    n_components : int, default=20
        Number of components for projection methods.
    k : int, default=10
        Number of neighbors to find/preserve.
    n_queries : int, default=500
        Number of query points to use.
    random_state : int, default=42
        Random seed.
    verbose : bool, default=True
        Print progress.

    Returns
    -------
    results : list[dict]
        List of result dictionaries for each method.
    """
    rng = np.random.default_rng(random_state)

    if X is None:
        if verbose:
            print("Generating synthetic data...")
        X = create_synthetic_data(random_state=random_state)

    n_samples = X.shape[0]
    query_indices = rng.choice(n_samples, size=min(n_queries, n_samples), replace=False)
    X_query = X[query_indices]

    if verbose:
        print("Computing ground truth neighbors...")
    nn_exact = NearestNeighbors(n_neighbors=k + 1, algorithm="brute")
    nn_exact.fit(X)
    _, true_neighbors_full = nn_exact.kneighbors(X_query)
    true_neighbors = true_neighbors_full[:, 1:]

    results: list[dict[str, Any]] = []

    methods = [
        (
            "RandomProjection",
            RandomProjectionANN(n_components=n_components, random_state=random_state),
        ),
        ("PCA", PCAAnn(n_components=n_components, random_state=random_state)),
        (
            "NeighborPreservingPP",
            NeighborPreservingANN(
                n_components=n_components,
                k=k,
                n_candidates=50,
                n_refinement_steps=30,
                n_backfit_iters=2,
                subsample_size=min(1000, n_samples),
                use_pca_init=True,
                random_state=random_state,
                verbose=False,
            ),
        ),
    ]

    for name, model in methods:
        if verbose:
            print(f"Evaluating {name}...")
        result = evaluate_method(model, X, X_query, true_neighbors, k, name)
        results.append(result)
        if verbose:
            print(
                f"  Recall@{k}: {result['recall@k']:.4f}, "
                f"Fit: {result['fit_time_s']:.2f}s, "
                f"Query: {result['query_time_s']:.4f}s"
            )

    return results


def run_full_evaluation(
    n_samples: int = 5000,
    n_features: int = 50,
    k_values: tuple[int, ...] = (5, 10, 20),
    n_component_values: tuple[int, ...] = (10, 20, 30),
    random_state: int = 42,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Run comprehensive evaluation across multiple k and n_components values.

    Parameters
    ----------
    n_samples : int, default=5000
        Number of samples in synthetic data.
    n_features : int, default=50
        Number of features.
    k_values : tuple[int], default=(5, 10, 20)
        Values of k to test.
    n_component_values : tuple[int], default=(10, 20, 30)
        Number of components to test.
    random_state : int, default=42
        Random seed.
    verbose : bool, default=True
        Print progress.

    Returns
    -------
    all_results : list[dict]
        All results across configurations.
    """
    if verbose:
        print(f"Creating synthetic data: {n_samples} samples, {n_features} features")
    X = create_synthetic_data(
        n_samples=n_samples,
        n_features=n_features,
        random_state=random_state,
    )

    all_results: list[dict[str, Any]] = []

    for k in k_values:
        for n_comp in n_component_values:
            if verbose:
                print(f"\n--- k={k}, n_components={n_comp} ---")
            results = run_comparison(
                X=X,
                n_components=n_comp,
                k=k,
                random_state=random_state,
                verbose=verbose,
            )
            for r in results:
                r["k"] = k
                r["n_components"] = n_comp
            all_results.extend(results)

    return all_results


if __name__ == "__main__":
    results = run_comparison(verbose=True)
    print("\nSummary:")
    for r in results:
        print(f"  {r['method']}: recall={r['recall@k']:.4f}")
