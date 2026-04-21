import importlib.util

import numpy as np
import pytest

if importlib.util.find_spec("torch") is None:
    pytest.skip("torch is not installed", allow_module_level=True)

from symmlearn.lattice._pg_image import PGImage
from symmlearn.maps._utils import get_ref_map, get_rot_maps


pytestmark = pytest.mark.torch


IMG_SIZE = 25
PATCH_SIZE = 9
N_MAX = 4


def make_test_image(seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((IMG_SIZE, IMG_SIZE), dtype=float)


def make_pgimage(seed=0):
    img = make_test_image(seed=seed)
    unit_cell_corners = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    return PGImage(pg_number=1, img=img, patch_size=PATCH_SIZE, unit_cell_corners=unit_cell_corners)


class TestMapUtils:
    def test_get_rot_maps_returns_expected_shape(self):
        img = make_test_image()
        rot_maps = get_rot_maps(img, patch_size=PATCH_SIZE, n_max=N_MAX)

        assert rot_maps.ndim == 3
        assert rot_maps.shape[1:] == img.shape
        assert rot_maps.shape[0] == 4

    def test_get_rot_maps_normalized_output_is_bounded(self):
        img = make_test_image()
        rot_maps = get_rot_maps(img, patch_size=PATCH_SIZE, n_max=N_MAX, normalize_output=True)

        assert np.all(rot_maps <= 1.0 + 1e-6)
        assert np.all(rot_maps >= -1.0 - 1e-6)

    def test_get_ref_map_without_angle_returns_none_theta_map(self):
        img = make_test_image()
        ref_map, theta_map = get_ref_map(img, patch_size=PATCH_SIZE, n_max=N_MAX, return_angle=False)

        assert ref_map.shape == img.shape
        assert theta_map is None

    def test_get_ref_map_with_angle_returns_theta_map(self):
        img = make_test_image()
        ref_map, theta_map = get_ref_map(img, patch_size=PATCH_SIZE, n_max=N_MAX, return_angle=True)

        assert ref_map.shape == img.shape
        assert theta_map.shape == img.shape


class TestPGImageComputeSymmMaps:
    def test_compute_symm_maps_without_angle_sets_angle_derived_maps_to_none(self):
        pgimage = make_pgimage()

        pgimage.compute_symm_maps(n_max=N_MAX, return_angle=False, crop=False)

        assert pgimage.rot_maps is not None
        assert pgimage.ref_map is not None
        assert pgimage.theta_map is None
        assert pgimage.sin_map is None
        assert pgimage.cos_map is None
        assert pgimage.has_symm_maps is True

    def test_compute_symm_maps_with_angle_populates_all_maps(self):
        pgimage = make_pgimage()

        pgimage.compute_symm_maps(n_max=N_MAX, return_angle=True, crop=False)

        assert pgimage.rot_maps is not None
        assert pgimage.ref_map is not None
        assert pgimage.theta_map is not None
        assert pgimage.sin_map is not None
        assert pgimage.cos_map is not None
        assert pgimage.theta_map.shape == pgimage.img.shape
        assert pgimage.sin_map.shape == pgimage.img.shape
        assert pgimage.cos_map.shape == pgimage.img.shape

    def test_compute_symm_maps_crop_uses_patch_size_margin(self):
        pgimage = make_pgimage()
        s = PATCH_SIZE // 2

        pgimage.compute_symm_maps(n_max=N_MAX, return_angle=True, crop=True)

        expected_shape = (IMG_SIZE - 2 * s, IMG_SIZE - 2 * s)
        assert pgimage.rot_maps.shape == (4, *expected_shape)
        assert pgimage.ref_map.shape == expected_shape
        assert pgimage.theta_map.shape == expected_shape
        assert pgimage.sin_map.shape == expected_shape
        assert pgimage.cos_map.shape == expected_shape
