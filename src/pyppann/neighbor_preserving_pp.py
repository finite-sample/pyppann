"""Neighbor-Preserving Projection Pursuit implementation.

This module implements a novel projection pursuit variant that directly optimizes
neighbor preservation (recall@k) instead of traditional PP objectives like
non-Gaussianity or reconstruction error.

The key insight: for approximate nearest neighbor search, we want projections
that preserve local neighborhoods, not projections that find "interesting"
structure in a statistical sense.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class NeighborPreservingPP:
    """Projection Pursuit that optimizes for neighbor preservation.

    This class learns low-dimensional projections that maximize the preservation
    of k-nearest neighbor relationships. Unlike traditional PP that finds
    "interesting" projections (high kurtosis, etc.), this directly optimizes
    what matters for ANN: keeping neighbors close after projection.

    Parameters
    ----------
    n_components : int, default=20
        Number of projection directions to learn.
    k : int, default=10
        Number of neighbors to preserve.
    n_candidates : int, default=50
        Number of candidate directions to evaluate per iteration (PCA + perturbations).
    n_refinement_steps : int, default=50
        Number of local optimization steps per projection.
    n_backfit_iters : int, default=3
        Number of backfitting iterations to refine all projections.
    subsample_size : int | None, default=None
        If set, subsample this many points for faster evaluation.
        None means use all points.
    use_pca_init : bool, default=True
        Whether to initialize with PCA directions.
    random_state : int | None, default=None
        Random seed for reproducibility.
    verbose : bool, default=False
        Whether to print progress information.
    """

    def __init__(
        self,
        n_components: int = 20,
        k: int = 10,
        n_candidates: int = 50,
        n_refinement_steps: int = 50,
        n_backfit_iters: int = 3,
        subsample_size: int | None = None,
        use_pca_init: bool = True,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        self.n_components = n_components
        self.k = k
        self.n_candidates = n_candidates
        self.n_refinement_steps = n_refinement_steps
        self.n_backfit_iters = n_backfit_iters
        self.subsample_size = subsample_size
        self.use_pca_init = use_pca_init
        self.random_state = random_state
        self.verbose = verbose

        self.projections_: NDArray[np.floating] | None = None
        self.scaler_ = StandardScaler()
        self._original_neighbors: NDArray[np.intp] | None = None
        self._kth_distances: NDArray[np.floating] | None = None
        self._rng: np.random.Generator | None = None

    def _compute_original_neighbors(
        self, X: NDArray[np.floating]
    ) -> tuple[NDArray[np.intp], NDArray[np.floating]]:
        """Compute k-NN graph in original space (one-time cost).

        Returns indices and distances to k-th neighbor for adaptive temperature.
        """
        nn = NearestNeighbors(n_neighbors=self.k + 1, algorithm="auto")
        nn.fit(X)
        distances, indices = nn.kneighbors(X)
        return (
            cast(NDArray[np.intp], indices[:, 1:]),
            cast(NDArray[np.floating], distances[:, -1]),
        )

    def _compute_adaptive_temperature(
        self,
        X_proj: NDArray[np.floating],
        sample_indices: NDArray[np.intp] | None = None,
    ) -> float:
        """Compute adaptive temperature based on projected k-th neighbor distances."""
        X_sample = X_proj[sample_indices] if sample_indices is not None else X_proj

        dists_sq = cdist(X_sample, X_proj, metric="sqeuclidean")

        n_sample = X_sample.shape[0]
        if sample_indices is not None:
            for i in range(n_sample):
                dists_sq[i, sample_indices[i]] = np.inf
        else:
            np.fill_diagonal(dists_sq, np.inf)

        kth_distances = np.partition(dists_sq, self.k, axis=1)[:, self.k]
        mean_kth_dist = float(np.mean(kth_distances))
        return max(mean_kth_dist / self.k, 1e-6)

    def _soft_recall(
        self,
        X: NDArray[np.floating],
        projection: NDArray[np.floating],
        original_neighbors: NDArray[np.intp],
        sample_indices: NDArray[np.intp] | None = None,
    ) -> float:
        """Compute soft recall - differentiable approximation of neighbor recall.

        Uses adaptive temperature based on actual k-th neighbor distances in
        projected space for more robust optimization.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Scaled input data.
        projection : ndarray of shape (n_features,) or (n_features, n_proj)
            Projection direction(s).
        original_neighbors : ndarray of shape (n_samples, k)
            Indices of true k-NN in original space.
        sample_indices : ndarray of shape (n_subsample,) | None
            If provided, only evaluate on these points for speed.

        Returns
        -------
        soft_recall : float
            Soft recall score in [0, 1]. Higher is better.
        """
        if projection.ndim == 1:
            projection = projection.reshape(-1, 1)

        X_proj = X @ projection

        temperature = self._compute_adaptive_temperature(X_proj, sample_indices)

        if sample_indices is not None:
            X_proj_sample = X_proj[sample_indices]
            neighbors_sample = original_neighbors[sample_indices]
        else:
            X_proj_sample = X_proj
            neighbors_sample = original_neighbors

        n_sample = X_proj_sample.shape[0]

        dists_sq = cdist(X_proj_sample, X_proj, metric="sqeuclidean")

        if sample_indices is not None:
            for i in range(n_sample):
                dists_sq[i, sample_indices[i]] = np.inf
        else:
            np.fill_diagonal(dists_sq, np.inf)

        neg_dists = -dists_sq / temperature
        neg_dists_max = neg_dists.max(axis=1, keepdims=True)
        exp_neg_dists = np.exp(neg_dists - neg_dists_max)
        softmax_weights = exp_neg_dists / exp_neg_dists.sum(axis=1, keepdims=True)

        row_indices = np.arange(n_sample)[:, np.newaxis]
        neighbor_weights = softmax_weights[row_indices, neighbors_sample]
        total_soft_recall = float(neighbor_weights.sum(axis=1).mean())

        return total_soft_recall

    def _hard_recall(
        self,
        X: NDArray[np.floating],
        projection: NDArray[np.floating],
        original_neighbors: NDArray[np.intp],
    ) -> float:
        """Compute exact recall@k for evaluation.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Scaled input data.
        projection : ndarray of shape (n_features,) or (n_features, n_proj)
            Projection direction(s).
        original_neighbors : ndarray of shape (n_samples, k)
            Indices of true k-NN in original space.

        Returns
        -------
        recall : float
            Recall@k score in [0, 1].
        """
        if projection.ndim == 1:
            projection = projection.reshape(-1, 1)

        X_proj = X @ projection

        nn = NearestNeighbors(n_neighbors=self.k + 1, algorithm="auto")
        nn.fit(X_proj)
        _, proj_neighbors = nn.kneighbors(X_proj)
        proj_neighbors = proj_neighbors[:, 1:]

        recall_sum = 0.0
        n_samples = X.shape[0]
        for i in range(n_samples):
            orig_set = set(original_neighbors[i])
            proj_set = set(proj_neighbors[i])
            recall_sum += len(orig_set & proj_set) / self.k

        return float(recall_sum / n_samples)

    def _generate_random_direction(self, n_features: int) -> NDArray[np.floating]:
        """Generate a random unit direction."""
        assert self._rng is not None
        direction = self._rng.standard_normal(n_features)
        return direction / np.linalg.norm(direction)

    def _generate_perturbed_direction(
        self, base: NDArray[np.floating], noise_scale: float = 0.3
    ) -> NDArray[np.floating]:
        """Generate a perturbation of a base direction."""
        assert self._rng is not None
        noise = self._rng.standard_normal(base.shape) * noise_scale
        perturbed = base + noise
        return cast("NDArray[np.floating]", perturbed / np.linalg.norm(perturbed))

    def _refine_projection(
        self,
        X: NDArray[np.floating],
        initial_projection: NDArray[np.floating],
        original_neighbors: NDArray[np.intp],
        sample_indices: NDArray[np.intp] | None,
        existing_projections: NDArray[np.floating] | None = None,
    ) -> NDArray[np.floating]:
        """Refine projection using local optimization."""

        def neg_soft_recall(proj_flat: NDArray[np.floating]) -> float:
            proj = proj_flat / np.linalg.norm(proj_flat)
            if existing_projections is not None:
                combined = np.column_stack([existing_projections, proj])
            else:
                combined = proj.reshape(-1, 1)
            return -self._soft_recall(X, combined, original_neighbors, sample_indices)

        result = minimize(
            neg_soft_recall,
            initial_projection,
            method="L-BFGS-B",
            options={"maxiter": self.n_refinement_steps},
        )

        refined: NDArray[np.floating] = result.x
        return refined / np.linalg.norm(refined)

    def _select_greedy_projection(
        self,
        X: NDArray[np.floating],
        existing_projections: NDArray[np.floating] | None,
        original_neighbors: NDArray[np.intp],
        sample_indices: NDArray[np.intp] | None,
        pca_directions: NDArray[np.floating] | None,
        component_idx: int,
    ) -> NDArray[np.floating]:
        """Select the next projection that maximizes marginal recall improvement."""
        n_features = X.shape[1]

        best_projection = None
        best_score = -np.inf

        candidates: list[NDArray[np.floating]] = []

        if pca_directions is not None and component_idx < pca_directions.shape[1]:
            pca_dir = pca_directions[:, component_idx]
            candidates.append(pca_dir)
            for _ in range(self.n_candidates // 3):
                candidates.append(self._generate_perturbed_direction(pca_dir, 0.2))
                candidates.append(self._generate_perturbed_direction(pca_dir, 0.5))

        remaining = self.n_candidates - len(candidates)
        for _ in range(remaining):
            candidates.append(self._generate_random_direction(n_features))

        for candidate in candidates:
            if existing_projections is not None:
                combined = np.column_stack([existing_projections, candidate])
            else:
                combined = candidate.reshape(-1, 1)

            score = self._soft_recall(X, combined, original_neighbors, sample_indices)

            if score > best_score:
                best_score = score
                best_projection = candidate

        assert best_projection is not None
        refined = self._refine_projection(
            X,
            best_projection,
            original_neighbors,
            sample_indices,
            existing_projections,
        )

        return refined

    def _backfit_projections(
        self,
        X: NDArray[np.floating],
        projections: NDArray[np.floating],
        original_neighbors: NDArray[np.intp],
        sample_indices: NDArray[np.intp] | None,
    ) -> NDArray[np.floating]:
        """Backfit: re-optimize each projection while holding others fixed."""
        projections = projections.copy()
        n_components = projections.shape[1]

        for backfit_iter in range(self.n_backfit_iters):
            if self.verbose:
                print(f"  Backfit iteration {backfit_iter + 1}/{self.n_backfit_iters}")

            for i in range(n_components):
                other_indices = [j for j in range(n_components) if j != i]
                other_projs = projections[:, other_indices] if other_indices else None

                refined_i = self._refine_projection(
                    X,
                    projections[:, i],
                    original_neighbors,
                    sample_indices,
                    other_projs,
                )
                projections[:, i] = refined_i

            if self.verbose:
                recall = self._hard_recall(X, projections, original_neighbors)
                print(f"    Recall@{self.k} after backfit: {recall:.4f}")

        return projections

    def fit(self, X: NDArray[np.floating]) -> NeighborPreservingPP:
        """Fit the neighbor-preserving projection pursuit model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : NeighborPreservingPP
            The fitted model.
        """
        self._rng = np.random.default_rng(self.random_state)

        X_scaled = self.scaler_.fit_transform(X)
        n_samples, n_features = X_scaled.shape

        if self.verbose:
            print(f"Computing k-NN graph in original space (k={self.k})...")
        self._original_neighbors, self._kth_distances = (
            self._compute_original_neighbors(X_scaled)
        )

        if self.subsample_size is not None and self.subsample_size < n_samples:
            sample_indices = self._rng.choice(
                n_samples, self.subsample_size, replace=False
            )
            sample_indices = sample_indices.astype(np.intp)
        else:
            sample_indices = None

        pca_directions: NDArray[np.floating] | None = None
        if self.use_pca_init:
            if self.verbose:
                print("Computing PCA initialization...")
            pca = PCA(n_components=min(self.n_components, n_features))
            pca.fit(X_scaled)
            pca_directions = pca.components_.T

        projections_list: list[NDArray[np.floating]] = []

        for i in range(self.n_components):
            if self.verbose:
                print(f"Learning projection {i + 1}/{self.n_components}...")

            existing = np.column_stack(projections_list) if projections_list else None

            new_proj = self._select_greedy_projection(
                X_scaled,
                existing,
                self._original_neighbors,
                sample_indices,
                pca_directions,
                i,
            )
            projections_list.append(new_proj)

            if self.verbose:
                current_projs = np.column_stack(projections_list)
                recall = self._hard_recall(
                    X_scaled, current_projs, self._original_neighbors
                )
                print(f"  Current recall@{self.k}: {recall:.4f}")

        self.projections_ = np.column_stack(projections_list)

        if self.n_backfit_iters > 0:
            if self.verbose:
                print("Backfitting projections...")
            self.projections_ = self._backfit_projections(
                X_scaled,
                self.projections_,
                self._original_neighbors,
                sample_indices,
            )

        return self

    def transform(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Project data onto learned directions.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_projected : ndarray of shape (n_samples, n_components)
            Projected data.
        """
        if self.projections_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X_scaled = cast(NDArray[np.floating], self.scaler_.transform(X))
        return X_scaled @ self.projections_

    def fit_transform(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Fit and transform in one step.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to fit and transform.

        Returns
        -------
        X_projected : ndarray of shape (n_samples, n_components)
            Projected data.
        """
        self.fit(X)
        return self.transform(X)

    def get_recall_at_k(self, X: NDArray[np.floating]) -> float:
        """Compute recall@k on the training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data (must match what was passed to fit).

        Returns
        -------
        recall : float
            Recall@k score.
        """
        if self.projections_ is None or self._original_neighbors is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X_scaled = cast(NDArray[np.floating], self.scaler_.transform(X))
        return self._hard_recall(X_scaled, self.projections_, self._original_neighbors)


class NeighborPreservingANN:
    """Approximate Nearest Neighbors using Neighbor-Preserving Projection Pursuit.

    This class uses neighbor-preserving projection pursuit to learn a
    low-dimensional representation that specifically preserves k-NN relationships,
    then performs nearest neighbor search in the projected space.

    Parameters
    ----------
    n_components : int, default=20
        Number of projection directions.
    k : int, default=10
        Number of neighbors to preserve during fitting.
    n_candidates : int, default=50
        Number of candidates per projection (PCA + perturbations).
    n_refinement_steps : int, default=50
        Local optimization steps per projection.
    n_backfit_iters : int, default=3
        Number of backfitting iterations.
    subsample_size : int | None, default=None
        Subsample size for faster fitting.
    use_pca_init : bool, default=True
        Whether to use PCA initialization.
    random_state : int | None, default=None
        Random seed.
    verbose : bool, default=False
        Print progress.
    """

    def __init__(
        self,
        n_components: int = 20,
        k: int = 10,
        n_candidates: int = 50,
        n_refinement_steps: int = 50,
        n_backfit_iters: int = 3,
        subsample_size: int | None = None,
        use_pca_init: bool = True,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        self.n_components = n_components
        self.k = k
        self.n_candidates = n_candidates
        self.n_refinement_steps = n_refinement_steps
        self.n_backfit_iters = n_backfit_iters
        self.subsample_size = subsample_size
        self.use_pca_init = use_pca_init
        self.random_state = random_state
        self.verbose = verbose

        self.nppp_: NeighborPreservingPP | None = None
        self.knn_: NearestNeighbors | None = None
        self.X_projected_: NDArray[np.floating] | None = None

    def fit(self, X: NDArray[np.floating], k: int = 10) -> NeighborPreservingANN:
        """Fit the model and build the nearest neighbor index.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        k : int, default=10
            Number of neighbors for the kNN index.

        Returns
        -------
        self : NeighborPreservingANN
            The fitted model.
        """
        self.nppp_ = NeighborPreservingPP(
            n_components=self.n_components,
            k=self.k,
            n_candidates=self.n_candidates,
            n_refinement_steps=self.n_refinement_steps,
            n_backfit_iters=self.n_backfit_iters,
            subsample_size=self.subsample_size,
            use_pca_init=self.use_pca_init,
            random_state=self.random_state,
            verbose=self.verbose,
        )

        self.X_projected_ = self.nppp_.fit_transform(X)

        self.knn_ = NearestNeighbors(n_neighbors=k, algorithm="brute")
        self.knn_.fit(self.X_projected_)

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
        if self.nppp_ is None or self.knn_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X_projected = self.nppp_.transform(X)
        result = self.knn_.kneighbors(X_projected, n_neighbors, return_distance)
        if return_distance:
            return cast(
                tuple[NDArray[np.floating], NDArray[np.intp]],
                result,
            )
        return cast(NDArray[np.intp], result)

    def get_training_recall(self, X: NDArray[np.floating]) -> float:
        """Get recall@k on training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        recall : float
            Recall@k score.
        """
        if self.nppp_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.nppp_.get_recall_at_k(X)
