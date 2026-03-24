"""Comparison utilities for PCA vs Projection Pursuit."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from pyppur.projection_pursuit import ProjectionPursuit
from pyppur.utils.metrics import evaluate_embedding
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from pyppann.data import create_synthetic_data


def compare_pca_with_projection_pursuit(
    data_type: str = "swiss_roll",
) -> dict[str, Any]:
    """Compare PCA with Projection Pursuit on synthetic data.

    Parameters
    ----------
    data_type : str, default='swiss_roll'
        Type of synthetic data. Options: 'swiss_roll', 's_curve', 'moons'.

    Returns
    -------
    summary : dict
        Dictionary containing comparison metrics for each method.
    """
    X, y = create_synthetic_data(data_type=data_type)

    scaler = StandardScaler()
    X_scaled: NDArray[np.floating] = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca: NDArray[np.floating] = pca.fit_transform(X_scaled)

    pp_dist = ProjectionPursuit(
        n_components=2,
        objective="distance_distortion",
        alpha=1.5,
        n_init=3,
        random_state=42,
    )
    X_pp_dist: NDArray[np.floating] = pp_dist.fit_transform(X_scaled)

    pp_recon = ProjectionPursuit(
        n_components=2,
        objective="reconstruction",
        alpha=1.5,
        n_init=3,
        random_state=42,
    )
    X_pp_recon: NDArray[np.floating] = pp_recon.fit_transform(X_scaled)

    metrics_pca = evaluate_embedding(X_scaled, X_pca, labels=y)
    metrics_dist = evaluate_embedding(X_scaled, X_pp_dist, labels=y)
    metrics_recon = evaluate_embedding(X_scaled, X_pp_recon, labels=y)

    return {
        "Dataset": data_type,
        "PCA_Trust": metrics_pca["trustworthiness"],
        "PCA_Distortion": metrics_pca["distance_distortion"],
        "PCA_Silhouette": metrics_pca["silhouette"],
        "PP_Distance_Trust": metrics_dist["trustworthiness"],
        "PP_Distance_Distortion": metrics_dist["distance_distortion"],
        "PP_Distance_Silhouette": metrics_dist["silhouette"],
        "PP_Recon_Trust": metrics_recon["trustworthiness"],
        "PP_Recon_Distortion": metrics_recon["distance_distortion"],
        "PP_Recon_Silhouette": metrics_recon["silhouette"],
    }
