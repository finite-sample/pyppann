"""Tests for pyppann."""

import numpy as np
import pytest

from pyppann import (
    ANNOYWrapper,
    NeighborPreservingANN,
    NeighborPreservingPP,
    PCAAnn,
    ProjectionPursuitANN,
    RandomProjectionANN,
    compute_recall,
)


def annoy_works() -> bool:
    """Check if annoy works correctly on this platform."""
    try:
        from annoy import AnnoyIndex

        t = AnnoyIndex(3, "euclidean")
        for i in range(10):
            t.add_item(i, [float(i), 0.0, 0.0])
        t.build(2)
        result = t.get_nns_by_vector([0, 0, 0], 5)
        return len(result) == 5
    except Exception:
        return False


ANNOY_BROKEN = not annoy_works()
ANNOY_SKIP_REASON = "annoy library has known issues on this platform (macOS ARM)"


@pytest.fixture
def sample_data() -> np.ndarray:
    """Generate sample data for testing."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((100, 10))


class TestProjectionPursuitANN:
    """Tests for ProjectionPursuitANN."""

    def test_fit(self, sample_data: np.ndarray) -> None:
        """Test that fit returns self and sets attributes."""
        model = ProjectionPursuitANN(r=2, stage_maxiter=10, backfit_maxiter=2)
        result = model.fit(sample_data, k=5)

        assert result is model
        assert model.ppr is not None
        assert model.knn is not None

    def test_kneighbors(self, sample_data: np.ndarray) -> None:
        """Test kneighbors returns correct shapes."""
        model = ProjectionPursuitANN(r=2, stage_maxiter=10, backfit_maxiter=2)
        model.fit(sample_data, k=5)

        query = sample_data[:3]
        distances, indices = model.kneighbors(query, n_neighbors=5)

        assert distances.shape == (3, 5)
        assert indices.shape == (3, 5)

    def test_kneighbors_without_distance(self, sample_data: np.ndarray) -> None:
        """Test kneighbors with return_distance=False."""
        model = ProjectionPursuitANN(r=2, stage_maxiter=10, backfit_maxiter=2)
        model.fit(sample_data, k=5)

        query = sample_data[:3]
        result = model.kneighbors(query, n_neighbors=5, return_distance=False)

        assert result.shape == (3, 5)

    def test_kneighbors_not_fitted(self, sample_data: np.ndarray) -> None:
        """Test that kneighbors raises error if not fitted."""
        model = ProjectionPursuitANN()

        with pytest.raises(ValueError, match="not fitted"):
            model.kneighbors(sample_data[:3])


@pytest.mark.skipif(ANNOY_BROKEN, reason=ANNOY_SKIP_REASON)
class TestANNOYWrapper:
    """Tests for ANNOYWrapper."""

    def test_fit(self, sample_data: np.ndarray) -> None:
        """Test that fit returns self and builds index."""
        model = ANNOYWrapper(n_trees=5)
        result = model.fit(sample_data)

        assert result is model
        assert model.index is not None
        assert model.dim == 10

    def test_kneighbors(self, sample_data: np.ndarray) -> None:
        """Test kneighbors returns correct shapes."""
        model = ANNOYWrapper(n_trees=5)
        model.fit(sample_data)

        query = sample_data[:3]
        distances, indices = model.kneighbors(query, n_neighbors=5)

        assert distances.shape == (3, 5)
        assert indices.shape == (3, 5)

    def test_kneighbors_without_distance(self, sample_data: np.ndarray) -> None:
        """Test kneighbors with return_distance=False."""
        model = ANNOYWrapper(n_trees=5)
        model.fit(sample_data)

        query = sample_data[:3]
        result = model.kneighbors(query, n_neighbors=5, return_distance=False)

        assert result.shape == (3, 5)

    def test_kneighbors_not_fitted(self, sample_data: np.ndarray) -> None:
        """Test that kneighbors raises error if not fitted."""
        model = ANNOYWrapper()

        with pytest.raises(ValueError, match="not built"):
            model.kneighbors(sample_data[:3])

    def test_different_metrics(self, sample_data: np.ndarray) -> None:
        """Test ANNOY with different distance metrics."""
        for metric in ["angular", "euclidean"]:
            model = ANNOYWrapper(n_trees=5, metric=metric)
            model.fit(sample_data)
            distances, indices = model.kneighbors(sample_data[:1], n_neighbors=3)
            assert distances.shape == (1, 3)
            assert indices.shape == (1, 3)


class TestANNOYWrapperBasic:
    """Basic tests for ANNOYWrapper that don't require full functionality."""

    def test_init(self) -> None:
        """Test ANNOYWrapper initialization."""
        model = ANNOYWrapper(n_trees=10, metric="euclidean")
        assert model.n_trees == 10
        assert model.metric == "euclidean"
        assert model.index is None
        assert model.dim is None

    def test_kneighbors_not_fitted(self, sample_data: np.ndarray) -> None:
        """Test that kneighbors raises error if not fitted."""
        model = ANNOYWrapper()

        with pytest.raises(ValueError, match="not built"):
            model.kneighbors(sample_data[:3])


class TestNeighborPreservingPP:
    """Tests for NeighborPreservingPP."""

    def test_fit(self, sample_data: np.ndarray) -> None:
        """Test that fit returns self and sets attributes."""
        model = NeighborPreservingPP(
            n_components=3,
            k=5,
            n_candidates=10,
            n_refinement_steps=5,
            n_backfit_iters=1,
        )
        result = model.fit(sample_data)

        assert result is model
        assert model.projections_ is not None
        assert model.projections_.shape == (10, 3)

    def test_transform(self, sample_data: np.ndarray) -> None:
        """Test transform returns correct shape."""
        model = NeighborPreservingPP(
            n_components=3,
            k=5,
            n_candidates=10,
            n_refinement_steps=5,
            n_backfit_iters=1,
        )
        model.fit(sample_data)

        X_transformed = model.transform(sample_data)
        assert X_transformed.shape == (100, 3)

    def test_fit_transform(self, sample_data: np.ndarray) -> None:
        """Test fit_transform returns correct shape."""
        model = NeighborPreservingPP(
            n_components=3,
            k=5,
            n_candidates=10,
            n_refinement_steps=5,
            n_backfit_iters=1,
        )

        X_transformed = model.fit_transform(sample_data)
        assert X_transformed.shape == (100, 3)

    def test_transform_not_fitted(self, sample_data: np.ndarray) -> None:
        """Test that transform raises error if not fitted."""
        model = NeighborPreservingPP()

        with pytest.raises(ValueError, match="not fitted"):
            model.transform(sample_data)

    def test_get_recall_at_k(self, sample_data: np.ndarray) -> None:
        """Test that get_recall_at_k returns a valid recall score."""
        model = NeighborPreservingPP(
            n_components=5,
            k=5,
            n_candidates=10,
            n_refinement_steps=5,
            n_backfit_iters=1,
        )
        model.fit(sample_data)

        recall = model.get_recall_at_k(sample_data)
        assert 0.0 <= recall <= 1.0

    def test_subsample(self, sample_data: np.ndarray) -> None:
        """Test fitting with subsampling."""
        model = NeighborPreservingPP(
            n_components=3, k=5, n_candidates=10, subsample_size=50, n_backfit_iters=1
        )
        model.fit(sample_data)

        assert model.projections_ is not None
        assert model.projections_.shape == (10, 3)

    def test_pca_init(self, sample_data: np.ndarray) -> None:
        """Test that PCA initialization works."""
        model = NeighborPreservingPP(
            n_components=3,
            k=5,
            n_candidates=10,
            n_refinement_steps=5,
            n_backfit_iters=1,
            use_pca_init=True,
        )
        model.fit(sample_data)
        assert model.projections_ is not None

    def test_no_pca_init(self, sample_data: np.ndarray) -> None:
        """Test fitting without PCA initialization."""
        model = NeighborPreservingPP(
            n_components=3,
            k=5,
            n_candidates=10,
            n_refinement_steps=5,
            n_backfit_iters=1,
            use_pca_init=False,
        )
        model.fit(sample_data)
        assert model.projections_ is not None


class TestNeighborPreservingANN:
    """Tests for NeighborPreservingANN."""

    def test_fit(self, sample_data: np.ndarray) -> None:
        """Test that fit returns self and sets attributes."""
        model = NeighborPreservingANN(
            n_components=3,
            k=5,
            n_candidates=10,
            n_refinement_steps=5,
            n_backfit_iters=1,
        )
        result = model.fit(sample_data, k=5)

        assert result is model
        assert model.nppp_ is not None
        assert model.knn_ is not None

    def test_kneighbors(self, sample_data: np.ndarray) -> None:
        """Test kneighbors returns correct shapes."""
        model = NeighborPreservingANN(
            n_components=3,
            k=5,
            n_candidates=10,
            n_refinement_steps=5,
            n_backfit_iters=1,
        )
        model.fit(sample_data, k=5)

        query = sample_data[:3]
        distances, indices = model.kneighbors(query, n_neighbors=5)

        assert distances.shape == (3, 5)
        assert indices.shape == (3, 5)

    def test_kneighbors_without_distance(self, sample_data: np.ndarray) -> None:
        """Test kneighbors with return_distance=False."""
        model = NeighborPreservingANN(
            n_components=3,
            k=5,
            n_candidates=10,
            n_refinement_steps=5,
            n_backfit_iters=1,
        )
        model.fit(sample_data, k=5)

        query = sample_data[:3]
        result = model.kneighbors(query, n_neighbors=5, return_distance=False)

        assert result.shape == (3, 5)

    def test_kneighbors_not_fitted(self, sample_data: np.ndarray) -> None:
        """Test that kneighbors raises error if not fitted."""
        model = NeighborPreservingANN()

        with pytest.raises(ValueError, match="not fitted"):
            model.kneighbors(sample_data[:3])

    def test_get_training_recall(self, sample_data: np.ndarray) -> None:
        """Test get_training_recall returns valid score."""
        model = NeighborPreservingANN(
            n_components=5,
            k=5,
            n_candidates=10,
            n_refinement_steps=5,
            n_backfit_iters=1,
        )
        model.fit(sample_data, k=5)

        recall = model.get_training_recall(sample_data)
        assert 0.0 <= recall <= 1.0


class TestRandomProjectionANN:
    """Tests for RandomProjectionANN."""

    def test_fit(self, sample_data: np.ndarray) -> None:
        """Test that fit returns self and sets attributes."""
        model = RandomProjectionANN(n_components=5)
        result = model.fit(sample_data, k=5)

        assert result is model
        assert model.projection_ is not None
        assert model.knn_ is not None

    def test_kneighbors(self, sample_data: np.ndarray) -> None:
        """Test kneighbors returns correct shapes."""
        model = RandomProjectionANN(n_components=5)
        model.fit(sample_data, k=5)

        query = sample_data[:3]
        distances, indices = model.kneighbors(query, n_neighbors=5)

        assert distances.shape == (3, 5)
        assert indices.shape == (3, 5)

    def test_kneighbors_not_fitted(self, sample_data: np.ndarray) -> None:
        """Test that kneighbors raises error if not fitted."""
        model = RandomProjectionANN()

        with pytest.raises(ValueError, match="not fitted"):
            model.kneighbors(sample_data[:3])

    def test_transform(self, sample_data: np.ndarray) -> None:
        """Test transform returns correct shape."""
        model = RandomProjectionANN(n_components=5)
        model.fit(sample_data, k=5)

        X_transformed = model.transform(sample_data)
        assert X_transformed.shape == (100, 5)


class TestPCAAnn:
    """Tests for PCAAnn."""

    def test_fit(self, sample_data: np.ndarray) -> None:
        """Test that fit returns self and sets attributes."""
        model = PCAAnn(n_components=5)
        result = model.fit(sample_data, k=5)

        assert result is model
        assert model.pca_ is not None
        assert model.knn_ is not None

    def test_kneighbors(self, sample_data: np.ndarray) -> None:
        """Test kneighbors returns correct shapes."""
        model = PCAAnn(n_components=5)
        model.fit(sample_data, k=5)

        query = sample_data[:3]
        distances, indices = model.kneighbors(query, n_neighbors=5)

        assert distances.shape == (3, 5)
        assert indices.shape == (3, 5)

    def test_kneighbors_not_fitted(self, sample_data: np.ndarray) -> None:
        """Test that kneighbors raises error if not fitted."""
        model = PCAAnn()

        with pytest.raises(ValueError, match="not fitted"):
            model.kneighbors(sample_data[:3])

    def test_transform(self, sample_data: np.ndarray) -> None:
        """Test transform returns correct shape."""
        model = PCAAnn(n_components=5)
        model.fit(sample_data, k=5)

        X_transformed = model.transform(sample_data)
        assert X_transformed.shape == (100, 5)


class TestComputeRecall:
    """Tests for compute_recall utility."""

    def test_perfect_recall(self) -> None:
        """Test that identical neighbors give recall=1."""
        neighbors = np.array([[0, 1, 2], [3, 4, 5]])
        recall = compute_recall(neighbors, neighbors)
        assert recall == 1.0

    def test_zero_recall(self) -> None:
        """Test that completely different neighbors give recall=0."""
        true_neighbors = np.array([[0, 1, 2], [3, 4, 5]])
        pred_neighbors = np.array([[6, 7, 8], [9, 10, 11]])
        recall = compute_recall(true_neighbors, pred_neighbors)
        assert recall == 0.0

    def test_partial_recall(self) -> None:
        """Test partial overlap."""
        true_neighbors = np.array([[0, 1, 2]])
        pred_neighbors = np.array([[0, 1, 9]])
        recall = compute_recall(true_neighbors, pred_neighbors)
        assert recall == pytest.approx(2 / 3)
