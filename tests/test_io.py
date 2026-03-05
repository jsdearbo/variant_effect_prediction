"""Tests for vcf_utils.io module."""

import os
import tempfile

import pandas as pd
import pytest

from vcf_utils.io import parse_vcf, write_vcf, vcf_record_to_dict


_SAMPLE_VCF = """##fileformat=VCFv4.2
##INFO=<ID=AC,Number=A,Type=Integer>
##INFO=<ID=AF,Number=A,Type=Float>
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO
chr1\t100\trs1\tA\tT\t30\tPASS\tAC=2;AF=0.01
chr1\t200\trs2\tG\tC,A\t50\tPASS\tAC=1,3;AF=0.005,0.015
chr2\t300\trs3\tAC\tA\t.\t.\t.
"""


class TestParseVcf:
    def test_basic_parsing(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".vcf", delete=False) as f:
            f.write(_SAMPLE_VCF)
            path = f.name
        try:
            df = parse_vcf(path, split_multiallelic=False)
            assert len(df) == 3
            assert df.iloc[0]["chrom"] == "chr1"
            assert df.iloc[0]["pos"] == 100
            assert df.iloc[0]["ref"] == "A"
        finally:
            os.unlink(path)

    def test_multiallelic_split(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".vcf", delete=False) as f:
            f.write(_SAMPLE_VCF)
            path = f.name
        try:
            df = parse_vcf(path, split_multiallelic=True)
            # rs2 (G -> C,A) should become 2 rows
            assert len(df) == 4
            alts = df[df["id"] == "rs2"]["alt"].tolist()
            assert "C" in alts
            assert "A" in alts
        finally:
            os.unlink(path)

    def test_info_field_extraction(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".vcf", delete=False) as f:
            f.write(_SAMPLE_VCF)
            path = f.name
        try:
            df = parse_vcf(path, split_multiallelic=False, info_fields=["AC", "AF"])
            assert "AC" in df.columns
            assert df.iloc[0]["AC"] == "2"
            assert df.iloc[0]["AF"] == "0.01"
        finally:
            os.unlink(path)

    def test_empty_vcf(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".vcf", delete=False) as f:
            f.write("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
            path = f.name
        try:
            df = parse_vcf(path)
            assert len(df) == 0
        finally:
            os.unlink(path)


class TestWriteVcf:
    def test_round_trip(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".vcf", delete=False) as f:
            f.write(_SAMPLE_VCF)
            path = f.name

        try:
            df = parse_vcf(path, split_multiallelic=False)

            out_path = path + ".out.vcf"
            write_vcf(df, out_path)
            df2 = parse_vcf(out_path, split_multiallelic=False)

            assert len(df2) == len(df)
            assert list(df2["chrom"]) == list(df["chrom"])
            assert list(df2["pos"]) == list(df["pos"])
        finally:
            os.unlink(path)
            if os.path.exists(out_path):
                os.unlink(out_path)


class TestVcfRecordToDict:
    def test_basic(self):
        rec = vcf_record_to_dict("chr1", 100, "A", "T")
        assert rec["chrom"] == "chr1"
        assert rec["pos"] == 100
        assert rec["ref"] == "A"
        assert rec["alt"] == "T"
        assert rec["filter"] == "PASS"
