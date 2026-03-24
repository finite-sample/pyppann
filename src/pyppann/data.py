"""Data loading and synthetic data generation utilities."""

from __future__ import annotations

import os
import urllib.request
import zipfile
from typing import cast

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import make_moons, make_s_curve, make_swiss_roll


def load_glove_embeddings(
    file_path: str = "glove.6B.50d.txt",
    limit: int = 10000,
) -> tuple[NDArray[np.floating], dict[str, int]]:
    """Load GloVe word embeddings from file.

    Downloads the embeddings if not present locally.

    Parameters
    ----------
    file_path : str, default='glove.6B.50d.txt'
        Path to the GloVe embeddings file.
    limit : int, default=10000
        Maximum number of embeddings to load.

    Returns
    -------
    embeddings : ndarray of shape (n_words, n_dims)
        Word embedding vectors.
    word_to_index : dict
        Mapping from words to their indices in the embeddings array.
    """
    print(f"Loading GloVe embeddings from {file_path}...")
    if not os.path.exists(file_path):
        url = "https://nlp.stanford.edu/data/glove.6B.zip"
        print(f"Downloading GloVe embeddings from {url}...")
        urllib.request.urlretrieve(url, "glove.6B.zip")
        with zipfile.ZipFile("glove.6B.zip", "r") as zip_ref:
            zip_ref.extractall(".")
        print("Download complete")

    word_to_index: dict[str, int] = {}
    embeddings: list[NDArray[np.floating]] = []
    with open(file_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            values = line.rstrip().split(" ")
            word = values[0]
            vector = np.array(values[1:], dtype="float32")
            word_to_index[word] = i
            embeddings.append(vector)
    return np.array(embeddings), word_to_index


def create_synthetic_data(
    data_type: str = "swiss_roll",
    n_samples: int = 1000,
    noise: float = 0.1,
    n_ambient_dims: int = 50,
) -> tuple[NDArray[np.floating], NDArray[np.intp]]:
    """Create synthetic nonlinear data embedded in high-dimensional space.

    Parameters
    ----------
    data_type : str, default='swiss_roll'
        Type of data to generate. Options: 'swiss_roll', 's_curve', 'moons'.
    n_samples : int, default=1000
        Number of samples to generate.
    noise : float, default=0.1
        Noise level.
    n_ambient_dims : int, default=50
        Total dimensionality of the output space.

    Returns
    -------
    X : ndarray of shape (n_samples, n_ambient_dims)
        Generated data.
    labels : ndarray of shape (n_samples,)
        Cluster labels (3 classes based on position along manifold).
    """
    if data_type == "swiss_roll":
        X, colors = make_swiss_roll(n_samples=n_samples, noise=noise, random_state=42)
    elif data_type == "s_curve":
        X, colors = make_s_curve(n_samples=n_samples, noise=noise, random_state=42)
    elif data_type == "moons":
        X, colors = make_moons(n_samples=n_samples, noise=noise, random_state=42)
        colors = colors.astype(float)
    else:
        raise ValueError(f"Unsupported data type: {data_type}")

    labels: NDArray[np.intp] = np.zeros(n_samples, dtype=np.intp)
    p33 = np.percentile(colors, 33)
    p66 = np.percentile(colors, 66)
    labels[(colors >= p33) & (colors < p66)] = 1
    labels[colors >= p66] = 2

    X_high = np.zeros((n_samples, n_ambient_dims))
    X_high[:, : X.shape[1]] = X
    rng = np.random.default_rng(42)
    X_high[:, X.shape[1] :] = noise * rng.standard_normal(
        (n_samples, n_ambient_dims - X.shape[1])
    )

    return X_high, cast(NDArray[np.intp], labels)
