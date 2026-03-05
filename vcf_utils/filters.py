"""
Variant filtering utilities.

Provides functions to filter a DataFrame of genetic variants by allele
type, length, base composition, and quality — the standard preprocessing
step before variant effect prediction.
"""

from typing import Optional

import pandas as pd

_STANDARD_BASES = set("ACGT")


def filter_variants(
    variants: pd.DataFrame,
    standard_bases: bool = True,
    max_insert_len: Optional[int] = 0,
    max_del_len: Optional[int] = 0,
    min_qual: Optional[float] = None,
    pass_only: bool = False,
    null_string: str = "-",
    inplace: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Filter a variant DataFrame by allele properties.

    Supports filtering by non-standard bases, insertion/deletion length,
    quality score, and FILTER status. Designed to run before prediction
    to ensure the model receives only supported variant types.

    Parameters
    ----------
    variants : pd.DataFrame
        Must contain columns ``ref`` and ``alt``.
    standard_bases : bool
        If True, drop variants whose alleles contain characters other
        than A, C, G, T.
    max_insert_len : int or None
        Maximum allowed insertion length (alt longer than ref).
        None disables this filter.
    max_del_len : int or None
        Maximum allowed deletion length (ref longer than alt).
        None disables this filter.
    min_qual : float or None
        Minimum QUAL score. Requires a ``qual`` column. None disables.
    pass_only : bool
        If True, keep only variants with FILTER == "PASS" or ".".
    null_string : str
        String representing absence of a base (e.g., "-" for deletions).
    inplace : bool
        If True, modify ``variants`` in place and return None.

    Returns
    -------
    pd.DataFrame or None
        Filtered DataFrame (if inplace=False), or None (if inplace=True).
    """
    drop_idxs = set()

    # Non-standard bases
    if standard_bases:
        for idx, row in variants.iterrows():
            alleles = set(row["ref"].upper() + row["alt"].upper()) - {null_string}
            if alleles - _STANDARD_BASES:
                drop_idxs.add(idx)

    # Allele lengths
    ref_len = variants["ref"].apply(lambda x: 0 if x == null_string else len(x))
    alt_len = variants["alt"].apply(lambda x: 0 if x == null_string else len(x))

    if max_insert_len is not None:
        insert_too_long = variants.index[(alt_len - ref_len) > max_insert_len]
        drop_idxs.update(insert_too_long)

    if max_del_len is not None:
        del_too_long = variants.index[(ref_len - alt_len) > max_del_len]
        drop_idxs.update(del_too_long)

    # Quality filter
    if min_qual is not None and "qual" in variants.columns:
        numeric_qual = pd.to_numeric(variants["qual"], errors="coerce")
        low_qual = variants.index[numeric_qual < min_qual]
        drop_idxs.update(low_qual)

    # PASS filter
    if pass_only and "filter" in variants.columns:
        not_pass = variants.index[
            ~variants["filter"].isin(["PASS", ".", "pass"])
        ]
        drop_idxs.update(not_pass)

    drop_list = list(drop_idxs)
    if inplace:
        variants.drop(index=drop_list, inplace=True)
        return None
    return variants.drop(index=drop_list).reset_index(drop=True)


def filter_to_snvs(variants: pd.DataFrame, inplace: bool = False) -> Optional[pd.DataFrame]:
    """
    Keep only single-nucleotide variants (SNVs).

    Filters to rows where both ref and alt are exactly 1 standard base.
    This is the most common filter for sequence-to-function model scoring
    since most models handle SNVs natively.

    Parameters
    ----------
    variants : pd.DataFrame
        Must contain columns ``ref`` and ``alt``.
    inplace : bool
        If True, modify ``variants`` in place.

    Returns
    -------
    pd.DataFrame or None
    """
    is_snv = (
        (variants["ref"].str.len() == 1)
        & (variants["alt"].str.len() == 1)
        & variants["ref"].str.upper().str.match(r"^[ACGT]$")
        & variants["alt"].str.upper().str.match(r"^[ACGT]$")
    )

    if inplace:
        variants.drop(index=variants.index[~is_snv], inplace=True)
        return None
    return variants[is_snv].reset_index(drop=True)
