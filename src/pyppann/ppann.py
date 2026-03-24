"""Projection Pursuit Approximate Nearest Neighbors implementation."""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from skpp import ProjectionPursuitRegressor


class ProjectionPursuitANN:
    """Approximate Nearest Neighbors using Projection Pursuit dimensionality reduction.

    This class uses projection pursuit regression to learn a low-dimensional
    representation of the data, then performs nearest neighbor search in the
    projected space.

    Parameters
    ----------
    r : int, default=3
        Number of projection directions (ridge functions).
    stage_maxiter : int, default=50
        Maximum iterations per stage in the projection pursuit optimization.
    backfit_maxiter : int, default=5
        Maximum backfitting iterations.
    tol : float, default=1e-4
        Tolerance for convergence.
    """

    def __init__(
        self,
        r: int = 3,
        stage_maxiter: int = 50,
        backfit_maxiter: int = 5,
        tol: float = 1e-4,
    ) -> None:
        self.r = r
        self.stage_maxiter = stage_maxiter
        self.backfit_maxiter = backfit_maxiter
        self.tol = tol
        self.ppr: ProjectionPursuitRegressor | None = None
        self.knn: NearestNeighbors | None = None
        self.scaler = StandardScaler()
        self._fallback_mode = False

    def fit(self, X: NDArray[np.floating], k: int = 10) -> ProjectionPursuitANN:
        """Fit the projection pursuit model and build the nearest neighbor index.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        k : int, default=10
            Number of neighbors to use for the internal kNN model.

        Returns
        -------
        self : ProjectionPursuitANN
            The fitted estimator.
        """
        self._fallback_mode = False
        X_scaled = self.scaler.fit_transform(X)

        n_samples = X_scaled.shape[0]
        max_targets = min(5, n_samples // 20)
        reference_indices = np.random.choice(n_samples, max_targets, replace=False)
        target = np.zeros((n_samples, max_targets))
        for i, ref_idx in enumerate(reference_indices):
            target[:, i] = np.linalg.norm(
                X_scaled - X_scaled[ref_idx].reshape(1, -1), axis=1
            )

        self.ppr = ProjectionPursuitRegressor(
            r=self.r,
            stage_maxiter=self.stage_maxiter,
            backfit_maxiter=self.backfit_maxiter,
            eps_stage=self.tol,
            opt_level="medium",
            random_state=42,
        )
        self.ppr.fit(X_scaled, target)

        X_projected = self.ppr.transform(X_scaled)

        self.knn = NearestNeighbors(n_neighbors=k, algorithm="brute")
        self.knn.fit(X_projected)

        return self

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
        n_neighbors : int, optional
            Number of neighbors to return. Defaults to the value used in fit.
        return_distance : bool, default=True
            Whether to return distances along with indices.

        Returns
        -------
        distances : ndarray of shape (n_queries, n_neighbors)
            Distances to the nearest neighbors. Only returned if return_distance=True.
        indices : ndarray of shape (n_queries, n_neighbors)
            Indices of the nearest neighbors.
        """
        if self.knn is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if self.ppr is not None and not self._fallback_mode:
            X_scaled = self.scaler.transform(X)
            X_projected = self.ppr.transform(X_scaled)
            result = self.knn.kneighbors(X_projected, n_neighbors, return_distance)
        else:
            result = self.knn.kneighbors(X, n_neighbors, return_distance)

        if return_distance:
            return cast(tuple[NDArray[np.floating], NDArray[np.intp]], result)
        return cast(NDArray[np.intp], result)
