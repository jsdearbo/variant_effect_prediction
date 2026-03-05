"""
Variant coordinate utilities.

Maps genetic variants to genomic intervals suitable for model input,
and validates reference alleles against a genome sequence. Supports
the coordinate systems used by Borzoi-class models (524,288 bp input
windows centered on variants).
"""

from typing import Optional

import numpy as np
import pandas as pd


def variants_to_intervals(
    variants: pd.DataFrame,
    seq_len: int,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Create genomic intervals centered on each variant.

    Each interval is exactly ``seq_len`` bp, centered on the variant
    position, suitable for extracting input sequences for a genomic
    foundation model.

    Parameters
    ----------
    variants : pd.DataFrame
        Must contain columns ``chrom`` and ``pos`` (1-based position).
    seq_len : int
        Length of the output genomic intervals.
    inplace : bool
        If True, add ``start`` and ``end`` columns to variants in place.

    Returns
    -------
    pd.DataFrame
        Columns: chrom, start, end (0-based, half-open).
    """
    half = int(np.ceil(seq_len / 2))
    starts = variants["pos"] - half
    ends = starts + seq_len

    if inplace:
        variants["start"] = starts
        variants["end"] = ends
        return variants
    return pd.DataFrame({
        "chrom": variants["chrom"],
        "start": starts,
        "end": ends,
    })


def check_reference(
    variants: pd.DataFrame,
    genome_fasta: str,
    null_string: str = "-",
    strict: bool = False,
) -> pd.DataFrame:
    """
    Validate that VCF reference alleles match the reference genome.

    Extracts the sequence at each variant position from the genome FASTA
    and compares it to the stated REF allele. Mismatches are flagged
    in a ``ref_match`` boolean column.

    Parameters
    ----------
    variants : pd.DataFrame
        Must contain columns ``chrom``, ``pos`` (1-based), and ``ref``.
    genome_fasta : str
        Path to an indexed FASTA file (.fai index must exist).
    null_string : str
        String used for absent bases (deletions).
    strict : bool
        If True, raise ValueError on any mismatch. If False, add a
        ``ref_match`` column and return.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with an added ``ref_match`` boolean column.

    Raises
    ------
    ValueError
        If ``strict=True`` and any reference alleles don't match.
    ImportError
        If pysam is not installed.
    """
    try:
        import pysam
    except ImportError:
        raise ImportError(
            "pysam is required for reference checking. "
            "Install with: pip install pysam"
        )

    fasta = pysam.FastaFile(genome_fasta)
    matches = []

    try:
        for _, row in variants.iterrows():
            ref = row["ref"]
            if ref == null_string:
                matches.append(True)
                continue

            chrom = str(row["chrom"])
            pos = int(row["pos"])
            ref_len = len(ref)

            # VCF is 1-based; pysam.fetch is 0-based half-open
            try:
                genome_seq = fasta.fetch(chrom, pos - 1, pos - 1 + ref_len).upper()
                matches.append(genome_seq == ref.upper())
            except (ValueError, KeyError):
                # Chromosome not found — try with/without chr prefix
                alt_chrom = ("chr" + chrom) if not chrom.startswith("chr") else chrom[3:]
                try:
                    genome_seq = fasta.fetch(alt_chrom, pos - 1, pos - 1 + ref_len).upper()
                    matches.append(genome_seq == ref.upper())
                except (ValueError, KeyError):
                    matches.append(False)
    finally:
        fasta.close()

    variants = variants.copy()
    variants["ref_match"] = matches

    if strict:
        n_mismatch = sum(1 for m in matches if not m)
        if n_mismatch > 0:
            bad_idxs = [i for i, m in enumerate(matches) if not m]
            raise ValueError(
                f"Reference allele mismatch at {n_mismatch} variants "
                f"(indices: {bad_idxs[:10]}{'...' if n_mismatch > 10 else ''})"
            )

    return variants


def variant_to_ref_alt_seqs(
    chrom: str,
    pos: int,
    ref: str,
    alt: str,
    genome_fasta: str,
    seq_len: int,
) -> tuple[str, str]:
    """
    Extract ref and alt sequences centered on a variant.

    Pulls a ``seq_len`` window from the genome centered on the variant
    position, then substitutes the alternate allele to create the alt
    sequence.

    Parameters
    ----------
    chrom : str
        Chromosome.
    pos : int
        1-based position.
    ref : str
        Reference allele.
    alt : str
        Alternate allele.
    genome_fasta : str
        Path to indexed FASTA.
    seq_len : int
        Length of output sequences.

    Returns
    -------
    tuple of (str, str)
        (reference_sequence, alternate_sequence), each of length seq_len.
    """
    try:
        import pysam
    except ImportError:
        raise ImportError("pysam is required. Install with: pip install pysam")

    fasta = pysam.FastaFile(genome_fasta)
    try:
        half = seq_len // 2
        # 0-based start for the window
        start = pos - 1 - half
        end = start + seq_len

        ref_seq = fasta.fetch(chrom, max(0, start), end).upper()

        # Pad if near chromosome boundary
        if start < 0:
            ref_seq = "N" * abs(start) + ref_seq
        if len(ref_seq) < seq_len:
            ref_seq = ref_seq + "N" * (seq_len - len(ref_seq))

        # Position of variant within the extracted window
        var_offset = half
        actual_ref = ref_seq[var_offset : var_offset + len(ref)]

        if actual_ref != ref.upper():
            import warnings
            warnings.warn(
                f"Reference mismatch at {chrom}:{pos} — "
                f"expected {ref}, found {actual_ref}"
            )

        # Create alt sequence by substitution
        alt_seq = ref_seq[:var_offset] + alt.upper() + ref_seq[var_offset + len(ref):]

        # For indels, trim or pad to maintain seq_len
        if len(alt_seq) > seq_len:
            excess = len(alt_seq) - seq_len
            alt_seq = alt_seq[excess // 2 : excess // 2 + seq_len]
        elif len(alt_seq) < seq_len:
            deficit = seq_len - len(alt_seq)
            alt_seq = alt_seq + "N" * deficit
    finally:
        fasta.close()

    return ref_seq[:seq_len], alt_seq[:seq_len]
