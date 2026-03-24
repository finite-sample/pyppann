"""Baseline ANN methods for comparison.

Implements simple baselines:
- RandomProjectionANN: Gaussian random projections + brute kNN
- PCAAnn: PCA dimensionality reduction + brute kNN
"""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection


class RandomProjectionANN:
    """Approximate Nearest Neighbors using Gaussian Random Projections.

    A simple baseline that projects data using random Gaussian projections
    and performs brute-force kNN in the reduced space.

    Parameters
    ----------
    n_components : int, default=20
        Number of dimensions to project to.
    random_state : int | None, default=None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_components: int = 20,
        random_state: int | None = None,
    ) -> None:
        self.n_components = n_components
        self.random_state = random_state

        self.projection_: GaussianRandomProjection | None = None
        self.scaler_ = StandardScaler()
        self.knn_: NearestNeighbors | None = None
        self.X_projected_: NDArray[np.floating] | None = None

    def fit(self, X: NDArray[np.floating], k: int = 10) -> RandomProjectionANN:
        """Fit the model and build the nearest neighbor index.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        k : int, default=10
            Number of neighbors for the kNN index.

        Returns
        -------
        self : RandomProjectionANN
            The fitted model.
        """
        X_scaled = self.scaler_.fit_transform(X)

        self.projection_ = GaussianRandomProjection(
            n_components=self.n_components,
            random_state=self.random_state,
        )
        self.X_projected_ = self.projection_.fit_transform(X_scaled)

        self.knn_ = NearestNeighbors(n_neighbors=k, algorithm="brute")
        self.knn_.fit(self.X_projected_)

        return self

    def transform(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Project data onto random directions.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_projected : ndarray of shape (n_samples, n_components)
            Projected data.
        """
        if self.projection_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X_scaled = cast(NDArray[np.floating], self.scaler_.transform(X))
        return cast(NDArray[np.floating], self.projection_.transform(X_scaled))

    def kneighbors(
        self,
        X: NDArray[np.floating],
        n_neighbors: int | None = None,
        return_distance: bool = True,
    ) -> tuple[NDArray[np.floating], NDArray[np.intp]] | NDArray[np.intp]:
        """Find the k nearest neighbors for query points.

        Parameters
        ----------
        X : ndarray of shape (n_queries, n_features)
            Query points.
        n_neighbors : int | None, default=None
            Number of neighbors. Uses fit k if None.
        return_distance : bool, default=True
            Whether to return distances.

        Returns
        -------
        distances : ndarray of shape (n_queries, n_neighbors)
            Distances to neighbors (if return_distance=True).
        indices : ndarray of shape (n_queries, n_neighbors)
            Indices of neighbors.
        """
        if self.projection_ is None or self.knn_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X_projected = self.transform(X)
        result = self.knn_.kneighbors(X_projected, n_neighbors, return_distance)
        if return_distance:
            return cast(
                tuple[NDArray[np.floating], NDArray[np.intp]],
                result,
            )
        return cast(NDArray[np.intp], result)


class PCAAnn:
    """Approximate Nearest Neighbors using PCA.

    A baseline that uses PCA for dimensionality reduction and performs
    brute-force kNN in the reduced space.

    Parameters
    ----------
    n_components : int, default=20
        Number of principal components to keep.
    random_state : int | None, default=None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_components: int = 20,
        random_state: int | None = None,
    ) -> None:
        self.n_components = n_components
        self.random_state = random_state

        self.pca_: PCA | None = None
        self.scaler_ = StandardScaler()
        self.knn_: NearestNeighbors | None = None
        self.X_projected_: NDArray[np.floating] | None = None

    def fit(self, X: NDArray[np.floating], k: int = 10) -> PCAAnn:
        """Fit the model and build the nearest neighbor index.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        k : int, default=10
            Number of neighbors for the kNN index.

        Returns
        -------
        self : PCAAnn
            The fitted model.
        """
        X_scaled = self.scaler_.fit_transform(X)

        self.pca_ = PCA(
            n_components=self.n_components,
            random_state=self.random_state,
        )
        self.X_projected_ = self.pca_.fit_transform(X_scaled)

        self.knn_ = NearestNeighbors(n_neighbors=k, algorithm="brute")
        self.knn_.fit(self.X_projected_)

        return self

    def transform(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Project data onto principal components.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_projected : ndarray of shape (n_samples, n_components)
            Projected data.
        """
        if self.pca_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X_scaled = cast(NDArray[np.floating], self.scaler_.transform(X))
        return cast(NDArray[np.floating], self.pca_.transform(X_scaled))

    def kneighbors(
        self,
        X: NDArray[np.floating],
        n_neighbors: int | None = None,
        return_distance: bool = True,
    ) -> tuple[NDArray[np.floating], NDArray[np.intp]] | NDArray[np.intp]:
        """Find the k nearest neighbors for query points.

        Parameters
        ----------
        X : ndarray of shape (n_queries, n_features)
            Query points.
        n_neighbors : int | None, default=None
            Number of neighbors. Uses fit k if None.
        return_distance : bool, default=True
            Whether to return distances.

        Returns
        -------
        distances : ndarray of shape (n_queries, n_neighbors)
            Distances to neighbors (if return_distance=True).
        indices : ndarray of shape (n_queries, n_neighbors)
            Indices of neighbors.
        """
        if self.pca_ is None or self.knn_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X_projected = self.transform(X)
        result = self.knn_.kneighbors(X_projected, n_neighbors, return_distance)
        if return_distance:
            return cast(
                tuple[NDArray[np.floating], NDArray[np.intp]],
                result,
            )
        return cast(NDArray[np.intp], result)
