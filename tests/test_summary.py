"""Tests for analysis.summary module."""

import numpy as np
import pandas as pd
import pytest

from analysis.summary import (
    rank_variants,
    summarize_effects,
    classify_effect_direction,
)


def _make_effects(n=10, seed=42):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "chrom": ["chr1"] * n,
        "pos": range(100, 100 + n),
        "ref": ["A"] * n,
        "alt": ["T"] * n,
        "effect_size": rng.normal(0, 1, size=n),
    })


class TestRankVariants:
    def test_descending_by_abs_effect(self):
        df = _make_effects()
        ranked = rank_variants(df)
        abs_effects = ranked["effect_size"].abs()
        assert (abs_effects.diff().dropna() <= 0).all()

    def test_top_n(self):
        df = _make_effects(n=20)
        ranked = rank_variants(df, top_n=5)
        assert len(ranked) == 5
        assert ranked.iloc[0]["rank"] == 1

    def test_nan_excluded(self):
        df = _make_effects(n=5)
        df.loc[2, "effect_size"] = np.nan
        ranked = rank_variants(df)
        assert len(ranked) == 4


class TestSummarizeEffects:
    def test_global_summary(self):
        df = _make_effects()
        summary = summarize_effects(df)
        assert "count" in summary.columns
        assert summary.iloc[0]["count"] == 10

    def test_group_by(self):
        df = _make_effects(n=10)
        df["chrom"] = ["chr1"] * 5 + ["chr2"] * 5
        summary = summarize_effects(df, group_by="chrom")
        assert len(summary) == 2
        assert set(summary["chrom"]) == {"chr1", "chr2"}

    def test_with_pvalue(self):
        df = _make_effects()
        df["pvalue"] = [0.01, 0.001, 0.1, 0.5, 0.04, 0.06, 0.002, 0.9, 0.03, 0.001]
        summary = summarize_effects(df)
        assert "n_significant_0.05" in summary.columns


class TestClassifyEffectDirection:
    def test_basic_classification(self):
        df = pd.DataFrame({
            "effect_size": [1.5, -0.8, 0.01, -2.0, 0.0],
        })
        result = classify_effect_direction(df, threshold=0.1)
        dirs = result["direction"].tolist()
        assert dirs[0] == "activating"
        assert dirs[1] == "repressing"
        assert dirs[2] == "neutral"
        assert dirs[3] == "repressing"
        assert dirs[4] == "neutral"
