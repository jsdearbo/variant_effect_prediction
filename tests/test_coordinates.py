"""Tests for vcf_utils.coordinates module."""

import numpy as np
import pandas as pd
import pytest

from vcf_utils.coordinates import variants_to_intervals


class TestVariantsToIntervals:
    def test_centering(self):
        variants = pd.DataFrame({"chrom": ["chr1"], "pos": [1000]})
        result = variants_to_intervals(variants, seq_len=100)
        assert result.iloc[0]["start"] == 1000 - 50
        assert result.iloc[0]["end"] == 1000 + 50

    def test_seq_len_preserved(self):
        variants = pd.DataFrame({
            "chrom": ["chr1", "chr2", "chrX"],
            "pos": [100, 50000, 1_000_000],
        })
        result = variants_to_intervals(variants, seq_len=524_288)
        widths = result["end"] - result["start"]
        assert (widths == 524_288).all()

    def test_inplace(self):
        variants = pd.DataFrame({"chrom": ["chr1"], "pos": [500]})
        result = variants_to_intervals(variants, seq_len=100, inplace=True)
        assert "start" in result.columns
        assert "end" in result.columns
        # Should be the same object
        assert result is variants

    def test_odd_seq_len(self):
        variants = pd.DataFrame({"chrom": ["chr1"], "pos": [1000]})
        result = variants_to_intervals(variants, seq_len=101)
        width = result.iloc[0]["end"] - result.iloc[0]["start"]
        assert width == 101
