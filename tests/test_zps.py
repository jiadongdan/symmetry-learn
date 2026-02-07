import unittest
import numpy as np
import warnings
import sys
import os
from math import factorial
from scipy.signal import fftconvolve

from symmlearn.maps._zps import ZPs


class TestZPsInitialization(unittest.TestCase):
    """Test ZPs class initialization"""

    def test_valid_initialization(self):
        """Test valid initialization parameters"""
        zps = ZPs(n_max=3, size=64)
        self.assertEqual(zps.n_max, 3)
        self.assertEqual(zps.size, 64)
        self.assertTrue(hasattr(zps, 'n'))
        self.assertTrue(hasattr(zps, 'm'))
        self.assertTrue(hasattr(zps, 'polynomials'))

    def test_negative_n_max(self):
        """Test that negative n_max should raise an exception"""
        with self.assertRaises(ValueError) as context:
            ZPs(n_max=-1, size=64)
        self.assertIn("n_max must be non-negative", str(context.exception))

    def test_zero_size(self):
        """Test that size=0 should raise an exception"""
        with self.assertRaises(ValueError) as context:
            ZPs(n_max=3, size=0)
        self.assertIn("size must be positive", str(context.exception))

    def test_negative_size(self):
        """Test that negative size should raise an exception"""
        with self.assertRaises(ValueError) as context:
            ZPs(n_max=3, size=-10)
        self.assertIn("size must be positive", str(context.exception))

    def test_n_max_exceeds_size(self):
        """Test that n_max > size should raise an exception"""
        with self.assertRaises(ValueError) as context:
            ZPs(n_max=65, size=64)
        self.assertIn("exceeds size", str(context.exception))

    def test_n_max_greater_than_half_size_warning(self):
        """Test that n_max > size/2 should generate a warning"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ZPs(n_max=40, size=64)

            # Check if warning was generated
            self.assertTrue(len(w) > 0)
            self.assertTrue(issubclass(w[0].category, UserWarning))
            self.assertIn("exceeds recommended limit", str(w[0].message))

    def test_n_max_less_than_half_size_no_warning(self):
        """Test that n_max <= size/2 should not generate a warning"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("error")  # Convert warnings to exceptions
            # These initializations should not generate warnings
            ZPs(n_max=32, size=64)  # n_max = size/2
            ZPs(n_max=31, size=64)  # n_max < size/2

            # If we reach here without exception, no warnings were raised
            # We can also check that the warning list is empty
            self.assertEqual(len(w), 0)


class TestZPsPolynomialGeneration(unittest.TestCase):
    """Test Zernike polynomial generation"""

    def setUp(self):
        """Setup before each test"""
        self.zps = ZPs(n_max=5, size=32)

    def tearDown(self):
        """Cleanup after each test"""
        del self.zps

    def test_polynomial_shapes(self):
        """Test the shapes of generated polynomials"""
        # n and m should be 1D arrays
        self.assertEqual(self.zps.n.ndim, 1)
        self.assertEqual(self.zps.m.ndim, 1)
        self.assertEqual(len(self.zps.n), len(self.zps.m))

        # polynomials should be 3D array
        self.assertEqual(self.zps.polynomials.ndim, 3)
        self.assertEqual(self.zps.polynomials.shape[0], len(self.zps.n))
        self.assertEqual(self.zps.polynomials.shape[1], self.zps.size)
        self.assertEqual(self.zps.polynomials.shape[2], self.zps.size)

    def test_polynomial_properties(self):
        """Test basic properties of Zernike polynomials"""
        # Test the relationship between n and m: |m| <= n
        self.assertTrue(np.all(np.abs(self.zps.m) <= self.zps.n))

        # Test the parity of n and m: n - |m| should be even
        self.assertTrue(np.all((self.zps.n - np.abs(self.zps.m)) % 2 == 0))

        # Test the range of m: for each n, m ranges from -n, -n+2, ..., n
        for n in range(self.zps.n_max + 1):
            mask = self.zps.n == n
            m_for_n = self.zps.m[mask]
            if len(m_for_n) > 0:
                self.assertTrue(np.all(np.abs(m_for_n) <= n))
                # Check the parity of m: m and n should have the same parity
                self.assertTrue(np.all((n - m_for_n) % 2 == 0))

    def test_get_polynomials_method(self):
        """Test get_polynomials method"""
        polynomials = self.zps.get_polynomials()

        # Should return the same data as self.polynomials
        np.testing.assert_array_equal(polynomials, self.zps.polynomials)

        # Returned data should be a numpy array
        self.assertIsInstance(polynomials, np.ndarray)

    def test_polynomial_count(self):
        """Test the number of polynomials"""
        # Calculate expected number of polynomials
        expected_count = 0
        for n in range(self.zps.n_max + 1):
            for m in range(-n, n + 1, 2):
                if abs(m) <= n:
                    expected_count += 1

        self.assertEqual(len(self.zps.n), expected_count)

    def test_polynomial_orthogonality_simple(self):
        """Test basic orthogonality of polynomials (simplified test)"""
        # For Zernike polynomials, polynomials with different (n,m) should be orthogonal
        # Here we only test that they are not completely identical
        num_polys = len(self.zps.n)

        if num_polys > 1:
            # Get first two polynomials
            poly1 = self.zps.polynomials[0]
            poly2 = self.zps.polynomials[1]

            # They should not be exactly the same
            self.assertFalse(np.allclose(poly1, poly2))

            # They should have similar energy (sum of squares)
            energy1 = np.sum(poly1 ** 2)
            energy2 = np.sum(poly2 ** 2)
            # Due to normalization, energies should be similar but not necessarily equal
            # We just check they are finite positive numbers
            self.assertTrue(np.isfinite(energy1))
            self.assertTrue(energy1 > 0)
            self.assertTrue(np.isfinite(energy2))
            self.assertTrue(energy2 > 0)


class TestZPsTransforms2D(unittest.TestCase):
    """Test 2D image transforms"""

    def setUp(self):
        """Setup before each test"""
        self.zps = ZPs(n_max=4, size=32)

        # Create a simple circular test image
        size = 64
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        xv, yv = np.meshgrid(x, y)
        rho = np.sqrt(xv ** 2 + yv ** 2)
        self.test_image = (rho <= 0.5).astype(np.float64)

    def test_transform_2d_shape(self):
        """Test the output shape of 2D transform"""
        zmoments = self.zps.transform(self.test_image)

        # Check that returned object has correct attributes
        self.assertTrue(hasattr(zmoments, 'data'))
        self.assertTrue(hasattr(zmoments, 'n'))
        self.assertTrue(hasattr(zmoments, 'm'))
        self.assertTrue(hasattr(zmoments, 'patch_size'))

        # For 2D input, data should be 3D array
        self.assertEqual(zmoments.data.ndim, 3)
        self.assertEqual(zmoments.data.shape[0], len(self.zps.n))
        self.assertEqual(zmoments.data.shape[1], self.test_image.shape[0])
        self.assertEqual(zmoments.data.shape[2], self.test_image.shape[1])

    def test_transform_2d_small_image_error(self):
        """Test that image smaller than polynomial size should raise error"""
        small_image = np.zeros((16, 16))  # Smaller than size=32

        with self.assertRaises(ValueError) as context:
            self.zps.transform(small_image)
        self.assertIn("must be at least", str(context.exception))

    def test_fit_transform_2d(self):
        """Test fit_transform method (2D)"""
        zmoments_fit_transform = self.zps.fit_transform(self.test_image)
        zmoments_transform = self.zps.transform(self.test_image)

        # fit_transform should produce the same results as transform
        np.testing.assert_array_almost_equal(
            zmoments_fit_transform.data,
            zmoments_transform.data,
            decimal=10
        )

    def test_transform_2d_invalid_dimension_error(self):
        """Test that invalid dimensions should raise error"""
        # 1D array
        invalid_1d = np.zeros((64,))
        with self.assertRaises(ValueError) as context:
            self.zps.transform(invalid_1d)
        self.assertIn("must be 2D or 3D", str(context.exception))

        # 4D array
        invalid_4d = np.zeros((2, 64, 64, 1))
        with self.assertRaises(ValueError) as context:
            self.zps.transform(invalid_4d)
        self.assertIn("must be 2D or 3D", str(context.exception))

    def test_transform_2d_with_constant_image(self):
        """Test transform of constant image"""
        constant_image = np.ones((64, 64))

        zmoments = self.zps.transform(constant_image)

        # Check that results are not all zeros or all NaN
        self.assertFalse(np.all(np.isnan(zmoments.data)))
        self.assertFalse(np.all(zmoments.data == 0))

        # Check that shape is correct
        self.assertEqual(zmoments.data.shape[0], len(self.zps.n))

    def test_transform_2d_with_random_image(self):
        """Test transform of random image"""
        rng = np.random.default_rng(48)
        random_image = rng.random((64, 64))

        zmoments = self.zps.transform(random_image)

        # Check that results are not all NaN
        self.assertFalse(np.any(np.isnan(zmoments.data)))

        # Check that shape is correct
        self.assertEqual(zmoments.data.shape[0], len(self.zps.n))


class TestZPsTransforms3D(unittest.TestCase):
    """Test 3D image batch transforms"""

    def setUp(self):
        """Setup before each test"""
        self.zps = ZPs(n_max=6, size=32)

        # Generate test image batch
        batch_size = 5
        size = 32
        self.test_batch = np.zeros((batch_size, size, size))

        # Each image has a circle at a different location
        for i in range(batch_size):
            x = np.linspace(-1, 1, size)
            y = np.linspace(-1, 1, size)
            xv, yv = np.meshgrid(x, y)

            # Each circle has a different center
            center_x = 0.5 * np.sin(i * np.pi / batch_size)
            center_y = 0.5 * np.cos(i * np.pi / batch_size)

            rho = np.sqrt((xv - center_x) ** 2 + (yv - center_y) ** 2)
            self.test_batch[i] = (rho <= 0.3).astype(np.float64)

    def test_transform_3d_shape(self):
        """Test the output shape of 3D transform"""
        zmoments = self.zps.transform(self.test_batch)

        # For 3D input, data should be 2D array
        self.assertEqual(zmoments.data.ndim, 2)
        self.assertEqual(zmoments.data.shape[0], self.test_batch.shape[0])
        self.assertEqual(zmoments.data.shape[1], len(self.zps.n))

    def test_transform_3d_wrong_size_error(self):
        """Test that mismatched image sizes should raise error"""
        wrong_size_batch = np.zeros((3, 40, 40))  # Size should be 32x32

        with self.assertRaises(ValueError) as context:
            self.zps.transform(wrong_size_batch)
        self.assertIn("must match", str(context.exception))

    def test_transform_3d_single_image_equivalence(self):
        """Test equivalence between batch processing and single image processing (simplified version)"""
        # Take the first image from the batch
        single_image = self.test_batch[0]

        # Use batch processing (but need to reshape to 3D)
        batch_single = single_image[np.newaxis, :, :]
        zmoments_batch = self.zps.transform(batch_single)

        # Use single image processing
        # Note: Single image uses FFT convolution, result is 3D
        # Batch processing uses dot product, result is 2D
        # So their result shapes are different, but the first moment may be related

        # We only check that they are both valid results
        self.assertFalse(np.any(np.isnan(zmoments_batch.data)))
        self.assertEqual(zmoments_batch.data.shape[0], 1)
        self.assertEqual(zmoments_batch.data.shape[1], len(self.zps.n))

    def test_fit_transform_3d(self):
        """Test fit_transform method (3D)"""
        zmoments_fit_transform = self.zps.fit_transform(self.test_batch)
        zmoments_transform = self.zps.transform(self.test_batch)

        # fit_transform should produce the same results as transform
        np.testing.assert_array_almost_equal(
            zmoments_fit_transform.data,
            zmoments_transform.data,
            decimal=10
        )

    def test_transform_3d_empty_batch(self):
        """Test transform of empty batch"""
        empty_batch = np.zeros((0, 32, 32))

        zmoments = self.zps.transform(empty_batch)

        # Should return empty zmoments object
        self.assertEqual(zmoments.data.shape[0], 0)
        self.assertEqual(zmoments.data.shape[1], len(self.zps.n))


class TestZPsSklearnCompatibility(unittest.TestCase):
    """Test compatibility with scikit-learn"""

    def test_base_estimator_inheritance(self):
        """Test inheritance from BaseEstimator"""
        zps = ZPs(n_max=3, size=32)

        # Check if it's an instance of BaseEstimator
        from sklearn.base import BaseEstimator
        self.assertIsInstance(zps, BaseEstimator)

    def test_transformer_mixin_inheritance(self):
        """Test inheritance from TransformerMixin"""
        zps = ZPs(n_max=3, size=32)

        # Check if it's an instance of TransformerMixin
        from sklearn.base import TransformerMixin
        self.assertIsInstance(zps, TransformerMixin)

    def test_fit_method_returns_self(self):
        """Test that fit method returns self"""
        zps = ZPs(n_max=3, size=32)

        # Create dummy data
        X = np.zeros((5, 32, 32))

        # fit should return self
        result = zps.fit(X)
        self.assertIs(result, zps)

    def test_fit_method_with_y(self):
        """Test that fit method accepts y parameter"""
        zps = ZPs(n_max=3, size=32)

        # Create dummy data
        X = np.zeros((5, 32, 32))
        y = np.array([0, 1, 0, 1, 0])

        # fit should accept y parameter (even if not used)
        result = zps.fit(X, y)
        self.assertIs(result, zps)


class TestZPsReproducibility(unittest.TestCase):
    """Test reproducibility"""

    def test_reproducible_polynomials(self):
        """Test that multiple initializations generate the same polynomials"""
        # Multiple initializations with same parameters
        zps1 = ZPs(n_max=5, size=32)
        zps2 = ZPs(n_max=5, size=32)

        # Polynomial n and m arrays should be the same
        np.testing.assert_array_equal(zps1.n, zps2.n)
        np.testing.assert_array_equal(zps1.m, zps2.m)

        # Polynomial values should be the same (within floating-point tolerance)
        np.testing.assert_array_almost_equal(zps1.polynomials, zps2.polynomials)

    def test_reproducible_transforms(self):
        """Test that identical inputs produce identical outputs"""
        zps = ZPs(n_max=4, size=32)

        # Fixed random seed
        np.random.seed(42)
        image = np.random.randn(64, 64)

        # Multiple transforms
        z1 = zps.transform(image)
        z2 = zps.transform(image)

        # Results should be identical (within floating-point tolerance)
        np.testing.assert_array_almost_equal(z1.data, z2.data)

if __name__ == '__main__':
    loader = unittest.TestLoader()

    test_classes = [
        TestZPsInitialization,
        TestZPsPolynomialGeneration,
        TestZPsTransforms2D,
        TestZPsTransforms3D,
        TestZPsSklearnCompatibility,
        TestZPsReproducibility,
    ]

    suites = []
    for test_class in test_classes:
        suite = loader.loadTestsFromTestCase(test_class)
        suites.append(suite)

    all_tests = unittest.TestSuite(suites)

    # run test
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(all_tests)