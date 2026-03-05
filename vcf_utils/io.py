"""
VCF file I/O utilities.

Reads VCF files into pandas DataFrames for downstream variant effect
prediction. Supports gzipped VCFs, multi-allelic splitting, and
round-trip writing.
"""

import gzip
import os
from typing import Optional

import pandas as pd


_VCF_COLUMNS = ["chrom", "pos", "id", "ref", "alt", "qual", "filter", "info"]


def _open_maybe_gzip(path: str):
    """Open a file, auto-detecting gzip compression."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def parse_vcf(
    vcf_path: str,
    split_multiallelic: bool = True,
    info_fields: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Parse a VCF file into a pandas DataFrame.

    Reads the 8 fixed VCF columns (CHROM through INFO). Multi-allelic
    records (comma-separated ALT) are optionally split into separate rows.
    Specific INFO sub-fields can be extracted into their own columns.

    Parameters
    ----------
    vcf_path : str
        Path to a VCF or VCF.gz file.
    split_multiallelic : bool
        If True, split multi-allelic ALT records into separate rows.
    info_fields : list of str, optional
        INFO sub-field keys to extract into separate columns (e.g. ["AC", "AF"]).

    Returns
    -------
    pd.DataFrame
        Columns: chrom, pos, id, ref, alt, qual, filter, info
        (plus any extracted info_fields).
    """
    rows = []
    with _open_maybe_gzip(vcf_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t", maxsplit=8)
            if len(fields) < 8:
                continue
            row = {
                "chrom": fields[0],
                "pos": int(fields[1]),
                "id": fields[2],
                "ref": fields[3],
                "alt": fields[4],
                "qual": fields[5],
                "filter": fields[6],
                "info": fields[7] if len(fields) > 7 else ".",
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=_VCF_COLUMNS)

    # Split multi-allelic
    if split_multiallelic:
        df = _split_multiallelic(df)

    # Extract INFO sub-fields
    if info_fields:
        df = _extract_info_fields(df, info_fields)

    df = df.reset_index(drop=True)
    return df


def _split_multiallelic(df: pd.DataFrame) -> pd.DataFrame:
    """Split comma-separated ALT alleles into separate rows."""
    mask = df["alt"].str.contains(",", na=False)
    if not mask.any():
        return df

    single = df[~mask].copy()
    multi = df[mask].copy()

    expanded = []
    for _, row in multi.iterrows():
        alts = row["alt"].split(",")
        for alt in alts:
            new_row = row.copy()
            new_row["alt"] = alt
            expanded.append(new_row)

    return pd.concat([single, pd.DataFrame(expanded)], ignore_index=True)


def _extract_info_fields(df: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    """Parse specific key=value pairs from the INFO column."""
    for field in fields:
        values = []
        for info_str in df["info"]:
            val = None
            for entry in info_str.split(";"):
                if entry.startswith(field + "="):
                    val = entry.split("=", 1)[1]
                    break
            values.append(val)
        df[field] = values
    return df


def vcf_record_to_dict(
    chrom: str, pos: int, ref: str, alt: str,
    variant_id: str = ".", qual: str = ".", filt: str = "PASS",
) -> dict:
    """
    Create a single variant record as a dict.

    Convenience function for programmatic variant creation without
    reading a VCF file.

    Parameters
    ----------
    chrom : str
        Chromosome (e.g., "chr1").
    pos : int
        1-based position.
    ref : str
        Reference allele.
    alt : str
        Alternate allele.
    variant_id : str
        Variant identifier (default ".").
    qual : str
        Quality score (default ".").
    filt : str
        Filter status (default "PASS").

    Returns
    -------
    dict
        Keys matching the VCF column names.
    """
    return {
        "chrom": chrom,
        "pos": pos,
        "id": variant_id,
        "ref": ref,
        "alt": alt,
        "qual": qual,
        "filter": filt,
        "info": ".",
    }


def write_vcf(df: pd.DataFrame, output_path: str, header_lines: Optional[list[str]] = None) -> None:
    """
    Write a variants DataFrame back to VCF format.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: chrom, pos, ref, alt.
    output_path : str
        Output file path.
    header_lines : list of str, optional
        Extra VCF header lines (without trailing newline). A minimal
        header is always included.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as fh:
        fh.write("##fileformat=VCFv4.2\n")
        if header_lines:
            for line in header_lines:
                fh.write(line.rstrip("\n") + "\n")
        fh.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")

        for _, row in df.iterrows():
            fields = [
                str(row.get("chrom", ".")),
                str(row.get("pos", ".")),
                str(row.get("id", ".")),
                str(row.get("ref", ".")),
                str(row.get("alt", ".")),
                str(row.get("qual", ".")),
                str(row.get("filter", ".")),
                str(row.get("info", ".")),
            ]
            fh.write("\t".join(fields) + "\n")
