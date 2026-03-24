"""ANNOY wrapper with sklearn-compatible API."""

from __future__ import annotations

from typing import Literal

import numpy as np
from annoy import AnnoyIndex
from numpy.typing import NDArray

AnnoyMetric = Literal["angular", "euclidean", "manhattan", "hamming", "dot"]


class ANNOYWrapper:
    """Wrapper around ANNOY with an sklearn-compatible API.

    Parameters
    ----------
    n_trees : int, default=10
        Number of trees in the forest. More trees gives higher accuracy.
    metric : str, default='angular'
        Distance metric. Options: 'angular', 'euclidean', 'manhattan', 'hamming', 'dot'.
    """

    def __init__(self, n_trees: int = 10, metric: AnnoyMetric = "angular") -> None:
        self.n_trees = n_trees
        self.metric: AnnoyMetric = metric
        self.index: AnnoyIndex | None = None
        self.dim: int | None = None
        self.data: NDArray[np.floating] | None = None

    def fit(self, X: NDArray[np.floating], k: int = 10) -> ANNOYWrapper:
        """Build the ANNOY index from training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        k : int, default=10
            Not used, included for API compatibility.

        Returns
        -------
        self : ANNOYWrapper
            The fitted estimator.
        """
        del k
        self.data = X
        self.dim = X.shape[1]

        self.index = AnnoyIndex(self.dim, self.metric)
        for i, x in enumerate(X):
            self.index.add_item(i, x)

        self.index.build(self.n_trees)
        return self

    def kneighbors(
        self,
        X: NDArray[np.floating],
        n_neighbors: int = 5,
        return_distance: bool = True,
    ) -> tuple[NDArray[np.floating], NDArray[np.intp]] | NDArray[np.intp]:
        """Find the k nearest neighbors for query points.

        Parameters
        ----------
        X : ndarray of shape (n_queries, n_features)
            Query points.
        n_neighbors : int, default=5
            Number of neighbors to return.
        return_distance : bool, default=True
            Whether to return distances along with indices.

        Returns
        -------
        distances : ndarray of shape (n_queries, n_neighbors)
            Distances to the nearest neighbors. Only returned if return_distance=True.
        indices : ndarray of shape (n_queries, n_neighbors)
            Indices of the nearest neighbors.
        """
        if self.index is None:
            raise ValueError("Index not built. Call fit() first.")

        indices_list = []
        distances_list = []
        for x in X:
            idx, dist = self.index.get_nns_by_vector(
                x, n_neighbors, include_distances=True
            )
            indices_list.append(idx)
            distances_list.append(dist)

        indices = np.array(indices_list, dtype=np.intp)
        if return_distance:
            distances = np.array(distances_list)
            return distances, indices
        return indices
