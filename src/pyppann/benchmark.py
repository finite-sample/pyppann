"""Benchmarking utilities for ANN methods."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors

from pyppann.annoy_wrapper import ANNOYWrapper
from pyppann.neighbor_preserving_pp import NeighborPreservingANN
from pyppann.ppann import ProjectionPursuitANN


def evaluate_similarity_search(
    embeddings: NDArray[np.floating],
    word_to_index: dict[str, int],
    index_to_word: dict[int, str],
    method: ProjectionPursuitANN | ANNOYWrapper | NearestNeighbors,
    n_queries: int = 10,
    k: int = 5,
) -> dict[str, dict[str, Any]]:
    """Evaluate semantic similarity search quality.

    Parameters
    ----------
    embeddings : ndarray of shape (n_words, n_dims)
        Word embeddings.
    word_to_index : dict
        Mapping from words to indices.
    index_to_word : dict
        Mapping from indices to words.
    method : estimator
        Fitted nearest neighbor method with kneighbors() method.
    n_queries : int, default=10
        Maximum number of semantic pairs to evaluate.
    k : int, default=5
        Number of neighbors to retrieve.

    Returns
    -------
    results : dict
        Dictionary mapping query words to their evaluation results.
    """
    semantic_pairs = [
        ("king", "queen"),
        ("man", "woman"),
        ("france", "paris"),
        ("japan", "tokyo"),
        ("computer", "laptop"),
        ("car", "vehicle"),
        ("happy", "sad"),
        ("river", "lake"),
        ("sun", "moon"),
        ("doctor", "hospital"),
    ]

    valid_pairs = [
        (w1, w2)
        for w1, w2 in semantic_pairs
        if w1 in word_to_index and w2 in word_to_index
    ][:n_queries]

    if not valid_pairs:
        print("No valid semantic pairs found in the vocabulary!")
        return {}

    results: dict[str, dict[str, Any]] = {}
    for w1, w2 in valid_pairs:
        idx1 = word_to_index[w1]
        query_vector = embeddings[idx1].reshape(1, -1)

        _, neighbor_indices = method.kneighbors(query_vector, n_neighbors=k + 1)
        neighbor_indices = neighbor_indices[0]

        neighbor_words = [index_to_word[idx] for idx in neighbor_indices[:k]]
        rank = -1
        for i, idx in enumerate(neighbor_indices):
            if index_to_word[idx] == w2:
                rank = i
                break

        results[w1] = {
            "query": w1,
            "expected": w2,
            "neighbors": neighbor_words,
            "found": w2 in neighbor_words,
            "rank": rank,
        }

    return results


def benchmark_ann_methods(
    embeddings: NDArray[np.floating],
    word_to_index: dict[str, int],
    random_seed: int = 42,
    n_queries: int = 1000,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, dict[str, Any]]]]:
    """Benchmark various ANN methods on word embeddings.

    Parameters
    ----------
    embeddings : ndarray of shape (n_words, n_dims)
        Word embeddings.
    word_to_index : dict
        Mapping from words to indices.
    random_seed : int, default=42
        Random seed for reproducibility.
    n_queries : int, default=1000
        Number of query points to benchmark.

    Returns
    -------
    results : dict
        Performance metrics for each method.
    semantic_results : dict
        Semantic search evaluation results for each method.
    """
    np.random.seed(random_seed)
    index_to_word = {idx: word for word, idx in word_to_index.items()}

    n_samples = embeddings.shape[0]
    X_train = embeddings
    k = 10

    results: dict[str, dict[str, float]] = {}
    semantic_results: dict[str, dict[str, dict[str, Any]]] = {}

    methods: dict[
        str,
        ProjectionPursuitANN | ANNOYWrapper | NeighborPreservingANN | NearestNeighbors,
    ] = {
        "PP-ANN (r=15)": ProjectionPursuitANN(
            r=15, stage_maxiter=500, backfit_maxiter=5
        ),
        "PP-ANN (r=30)": ProjectionPursuitANN(
            r=30, stage_maxiter=500, backfit_maxiter=5
        ),
        "NP-ANN (c=10)": NeighborPreservingANN(
            n_components=10, k=k, n_candidates=30, subsample_size=500
        ),
        "NP-ANN (c=15)": NeighborPreservingANN(
            n_components=15, k=k, n_candidates=30, subsample_size=500
        ),
        "ANNOY (10 trees)": ANNOYWrapper(n_trees=10, metric="angular"),
        "ANNOY (50 trees)": ANNOYWrapper(n_trees=50, metric="angular"),
        "ANNOY (100 trees)": ANNOYWrapper(n_trees=100, metric="angular"),
        "Exact NN": NearestNeighbors(n_neighbors=k, algorithm="brute"),
    }

    query_indices = np.random.choice(n_samples, n_queries, replace=False)
    X_query = X_train[query_indices]

    print("Building indices for all methods...")
    for name, method in methods.items():
        print(f"  Building index for {name}...")
        method.fit(X_train, k)

    chunk_size = 200

    for name, method in methods.items():
        print(f"\nBenchmarking query performance for {name}...")

        t0 = time.time()
        method.fit(X_train, k)
        fit_time = time.time() - t0

        all_indices = []
        query_start = time.time()
        for i in range(0, n_queries, chunk_size):
            end_idx = min(i + chunk_size, n_queries)
            chunk = X_query[i:end_idx]
            _, indices = method.kneighbors(chunk, n_neighbors=k)
            all_indices.append(indices)
        query_time = time.time() - query_start
        indices = np.vstack(all_indices)
        avg_query_time = query_time / n_queries

        if name != "Exact NN":
            nn = NearestNeighbors(n_neighbors=k, algorithm="brute")
            nn.fit(X_train)
            recall_sum = 0.0
            for i in range(0, n_queries, chunk_size):
                end_idx = min(i + chunk_size, n_queries)
                chunk = X_query[i:end_idx]
                _, exact_indices = nn.kneighbors(chunk, n_neighbors=k)
                for j in range(end_idx - i):
                    method_neighbors = set(indices[i + j])
                    exact_neighbors = set(exact_indices[j])
                    recall_sum += len(method_neighbors & exact_neighbors) / k
            recall = recall_sum / n_queries
        else:
            recall = 1.0

        semantic_eval = evaluate_similarity_search(
            embeddings, word_to_index, index_to_word, method, n_queries=15
        )
        semantic_accuracy = (
            sum(result["found"] for result in semantic_eval.values())
            / len(semantic_eval)
            if semantic_eval
            else 0.0
        )

        semantic_results[name] = semantic_eval
        results[name] = {
            "fit_time": fit_time,
            "query_time": avg_query_time,
            "recall@k": recall,
            "semantic_accuracy": semantic_accuracy,
        }

    return results, semantic_results
