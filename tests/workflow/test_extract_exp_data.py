# CI smoke test
import numpy as np
import pytest
from symmlearn.workflow._extract_features import extract_exp_data, ExtractionResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

IMG_H, IMG_W = 256, 256
PATCH_SIZE = 32
NUM_PATCHES = 20


@pytest.fixture
def gray_img():
    rng = np.random.default_rng(0)
    return rng.random((IMG_H, IMG_W))


@pytest.fixture
def multichannel_img():
    rng = np.random.default_rng(0)
    return rng.random((IMG_H, IMG_W, 4))


@pytest.fixture
def lbs_img():
    """Label image with 3 classes."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 3, size=(IMG_H, IMG_W))


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

class TestReturnType:
    def test_returns_extraction_result(self, gray_img):
        result = extract_exp_data(gray_img, PATCH_SIZE, NUM_PATCHES, seed=0)
        assert isinstance(result, ExtractionResult)

    def test_named_fields_accessible(self, gray_img):
        result = extract_exp_data(gray_img, PATCH_SIZE, NUM_PATCHES, seed=0)
        assert hasattr(result, 'pts')
        assert hasattr(result, 'patches')
        assert hasattr(result, 'lbs')


# ---------------------------------------------------------------------------
# Output shapes
# ---------------------------------------------------------------------------

class TestOutputShapes:
    def test_pts_shape(self, gray_img):
        result = extract_exp_data(gray_img, PATCH_SIZE, NUM_PATCHES, seed=0)
        assert result.pts.ndim == 2
        assert result.pts.shape[1] == 2

    def test_patches_shape_2d_image(self, gray_img):
        result = extract_exp_data(gray_img, PATCH_SIZE, NUM_PATCHES, seed=0)
        n = len(result.pts)
        assert result.patches.shape == (n, PATCH_SIZE, PATCH_SIZE)

    def test_patches_shape_multichannel_image(self, multichannel_img):
        result = extract_exp_data(multichannel_img, PATCH_SIZE, NUM_PATCHES, seed=0)
        n = len(result.pts)
        assert result.patches.shape == (n, PATCH_SIZE, PATCH_SIZE, 4)

    def test_pts_and_patches_same_length(self, gray_img):
        result = extract_exp_data(gray_img, PATCH_SIZE, NUM_PATCHES, seed=0)
        assert len(result.pts) == len(result.patches)

    def test_num_patches_respected(self, gray_img):
        result = extract_exp_data(gray_img, PATCH_SIZE, NUM_PATCHES, seed=0)
        assert len(result.pts) <= NUM_PATCHES


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

class TestLabels:
    def test_lbs_none_without_lbs_img(self, gray_img):
        result = extract_exp_data(gray_img, PATCH_SIZE, NUM_PATCHES, seed=0)
        assert result.lbs is None

    def test_lbs_shape_with_lbs_img(self, gray_img, lbs_img):
        result = extract_exp_data(gray_img, PATCH_SIZE, NUM_PATCHES, lbs_img=lbs_img, seed=0)
        assert result.lbs is not None
        assert result.lbs.shape == (len(result.pts),)

    def test_lbs_values_are_valid_labels(self, gray_img, lbs_img):
        valid_labels = set(np.unique(lbs_img))
        result = extract_exp_data(gray_img, PATCH_SIZE, NUM_PATCHES, lbs_img=lbs_img, seed=0)
        assert set(result.lbs).issubset(valid_labels)


# ---------------------------------------------------------------------------
# Boundary safety
# ---------------------------------------------------------------------------

class TestBoundarySafety:
    def test_patches_do_not_exceed_image_bounds(self, gray_img):
        half = PATCH_SIZE // 2
        result = extract_exp_data(gray_img, PATCH_SIZE, NUM_PATCHES, seed=0)
        xs, ys = result.pts[:, 0], result.pts[:, 1]
        assert np.all(xs - half >= 0)
        assert np.all(xs + half <= IMG_W)
        assert np.all(ys - half >= 0)
        assert np.all(ys + half <= IMG_H)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_gives_same_pts(self, gray_img):
        r1 = extract_exp_data(gray_img, PATCH_SIZE, NUM_PATCHES, seed=42)
        r2 = extract_exp_data(gray_img, PATCH_SIZE, NUM_PATCHES, seed=42)
        np.testing.assert_array_equal(r1.pts, r2.pts)

    def test_different_seeds_give_different_pts(self, gray_img):
        r1 = extract_exp_data(gray_img, PATCH_SIZE, NUM_PATCHES, seed=1)
        r2 = extract_exp_data(gray_img, PATCH_SIZE, NUM_PATCHES, seed=2)
        assert not np.array_equal(r1.pts, r2.pts)
