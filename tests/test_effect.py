"""Tests for prediction.effect module."""

import numpy as np
import pytest

from prediction.effect import compare_predictions


class TestComparePredictions:
    def test_log2fc_basic(self):
        ref = np.array([1.0, 2.0, 4.0])
        alt = np.array([2.0, 2.0, 1.0])
        result = compare_predictions(ref, alt, method="log2fc")
        np.testing.assert_allclose(result, [1.0, 0.0, -2.0], atol=1e-5)

    def test_subtract(self):
        ref = np.array([10.0, 20.0])
        alt = np.array([15.0, 10.0])
        result = compare_predictions(ref, alt, method="subtract")
        np.testing.assert_allclose(result, [5.0, -10.0])

    def test_divide(self):
        ref = np.array([2.0, 5.0])
        alt = np.array([4.0, 5.0])
        result = compare_predictions(ref, alt, method="divide")
        np.testing.assert_allclose(result, [2.0, 1.0])

    def test_log2fc_avoids_zero_division(self):
        ref = np.array([0.0, 1.0])
        alt = np.array([1.0, 0.0])
        result = compare_predictions(ref, alt, method="log2fc")
        # Should not raise or produce inf
        assert np.all(np.isfinite(result))

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown comparison"):
            compare_predictions(np.array([1.0]), np.array([1.0]), method="magic")

    def test_2d_input(self):
        ref = np.array([[1.0, 2.0], [4.0, 8.0]])
        alt = np.array([[2.0, 4.0], [4.0, 4.0]])
        result = compare_predictions(ref, alt, method="log2fc")
        assert result.shape == (2, 2)
        np.testing.assert_allclose(result[0], [1.0, 1.0], atol=1e-5)
        np.testing.assert_allclose(result[1], [0.0, -1.0], atol=1e-5)
