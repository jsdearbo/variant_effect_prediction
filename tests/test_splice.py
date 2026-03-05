"""Tests for prediction.splice module."""

import numpy as np
import pytest

from prediction.splice import one_hot_encode


class TestOneHotEncode:
    def test_forward_strand(self):
        enc = one_hot_encode("ACGT", strand="+")
        assert enc.shape == (4, 4)
        # A at position 0
        np.testing.assert_array_equal(enc[:, 0], [1, 0, 0, 0])
        # C at position 1
        np.testing.assert_array_equal(enc[:, 1], [0, 1, 0, 0])
        # G at position 2
        np.testing.assert_array_equal(enc[:, 2], [0, 0, 1, 0])
        # T at position 3
        np.testing.assert_array_equal(enc[:, 3], [0, 0, 0, 1])

    def test_reverse_strand(self):
        # RC of ACGT is ACGT reversed → TGCA, then complement → ACGT
        # Actually: reverse of "ACGT" = "TGCA", complement of TGCA = ACGT
        enc_fwd = one_hot_encode("ACGT", strand="+")
        enc_rev = one_hot_encode("ACGT", strand="-")
        # RC should equal encoding "ACGT" forward
        # "ACGT" reversed = "TGCA"
        # complement: T->A, G->C, C->G, A->T = "ACGT"
        # So enc_rev should be one_hot of "ACGT"
        assert enc_rev.shape == (4, 4)
        np.testing.assert_array_equal(enc_rev, enc_fwd)

    def test_n_encoded_as_zeros(self):
        enc = one_hot_encode("N", strand="+")
        np.testing.assert_array_equal(enc[:, 0], [0, 0, 0, 0])

    def test_output_shape(self):
        enc = one_hot_encode("ACGTACGTNN", strand="+")
        assert enc.shape == (4, 10)

    def test_invalid_strand_raises(self):
        with pytest.raises(ValueError, match="strand"):
            one_hot_encode("ACGT", strand="x")
