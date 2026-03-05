"""Tests for vcf_utils.filters module."""

import pandas as pd
import pytest

from vcf_utils.filters import filter_variants, filter_to_snvs


def _make_variants(*rows):
    """Helper: create a variant DataFrame from (ref, alt, ...) tuples."""
    records = []
    for row in rows:
        rec = {"chrom": "chr1", "pos": 100, "ref": row[0], "alt": row[1]}
        if len(row) > 2:
            rec["qual"] = row[2]
        if len(row) > 3:
            rec["filter"] = row[3]
        records.append(rec)
    return pd.DataFrame(records)


class TestFilterVariants:
    def test_standard_bases_drops_non_acgt(self):
        df = _make_variants(("A", "T"), ("A", "N"), ("R", "T"))
        result = filter_variants(df, standard_bases=True)
        assert len(result) == 1
        assert result.iloc[0]["alt"] == "T"

    def test_max_insert_len(self):
        df = _make_variants(("A", "T"), ("A", "ACGT"), ("A", "AC"))
        result = filter_variants(df, standard_bases=False, max_insert_len=1)
        assert len(result) == 2  # SNV + 1bp insert kept, 3bp insert dropped

    def test_max_del_len(self):
        df = _make_variants(("ACGT", "A"), ("AC", "A"), ("A", "T"))
        result = filter_variants(df, standard_bases=False, max_del_len=1)
        assert len(result) == 2  # 1bp del + SNV kept, 3bp del dropped

    def test_pass_only(self):
        df = _make_variants(("A", "T", ".", "PASS"), ("A", "C", ".", "LowQual"))
        result = filter_variants(df, standard_bases=False, pass_only=True, max_insert_len=None, max_del_len=None)
        assert len(result) == 1
        assert result.iloc[0]["alt"] == "T"

    def test_min_qual(self):
        df = _make_variants(("A", "T", "30", "."), ("A", "C", "10", "."))
        result = filter_variants(df, standard_bases=False, min_qual=20, max_insert_len=None, max_del_len=None)
        assert len(result) == 1
        assert result.iloc[0]["alt"] == "T"

    def test_inplace(self):
        df = _make_variants(("A", "T"), ("A", "N"))
        result = filter_variants(df, standard_bases=True, inplace=True)
        assert result is None
        assert len(df) == 1

    def test_empty_input(self):
        df = pd.DataFrame(columns=["chrom", "pos", "ref", "alt"])
        result = filter_variants(df)
        assert len(result) == 0


class TestFilterToSnvs:
    def test_keeps_only_snvs(self):
        df = _make_variants(("A", "T"), ("AC", "A"), ("A", "GC"), ("G", "C"))
        result = filter_to_snvs(df)
        assert len(result) == 2
        assert list(result["ref"]) == ["A", "G"]

    def test_inplace(self):
        df = _make_variants(("A", "T"), ("AC", "A"))
        filter_to_snvs(df, inplace=True)
        assert len(df) == 1
