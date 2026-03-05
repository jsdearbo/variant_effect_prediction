"""
Variant effect summary and annotation utilities.

Provides functions to rank variants by effect size, compute summary
statistics, and annotate variants with genomic context (gene overlap,
regulatory region, etc.).
"""

from typing import Optional

import numpy as np
import pandas as pd


def rank_variants(
    effects: pd.DataFrame,
    score_col: str = "effect_size",
    ascending: bool = False,
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    """
    Rank variants by effect size.

    Parameters
    ----------
    effects : pd.DataFrame
        Variant effect DataFrame (output of ``predict_variant_effects``
        or ``marginalize_variants``).
    score_col : str
        Column to rank by.
    ascending : bool
        Sort order. False = largest effects first.
    top_n : int, optional
        Return only the top N variants.

    Returns
    -------
    pd.DataFrame
        Sorted DataFrame with an added ``rank`` column.
    """
    df = effects.copy()
    df = df.dropna(subset=[score_col])
    df["abs_effect"] = df[score_col].abs()
    df = df.sort_values("abs_effect", ascending=ascending).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    df = df.drop(columns=["abs_effect"])

    if top_n is not None:
        df = df.head(top_n)

    return df


def summarize_effects(
    effects: pd.DataFrame,
    score_col: str = "effect_size",
    group_by: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute summary statistics for variant effects.

    Parameters
    ----------
    effects : pd.DataFrame
        Variant effect DataFrame.
    score_col : str
        Column containing effect sizes.
    group_by : str, optional
        Column to group by (e.g., "chrom", "gene") before computing
        statistics. If None, returns a single-row summary.

    Returns
    -------
    pd.DataFrame
        Summary statistics: count, mean, std, min, max, median,
        n_significant (if pvalue column exists).
    """
    def _stats(group):
        scores = group[score_col].dropna()
        row = {
            "count": len(scores),
            "mean": scores.mean() if len(scores) > 0 else np.nan,
            "std": scores.std() if len(scores) > 1 else np.nan,
            "min": scores.min() if len(scores) > 0 else np.nan,
            "max": scores.max() if len(scores) > 0 else np.nan,
            "median": scores.median() if len(scores) > 0 else np.nan,
            "mean_abs": scores.abs().mean() if len(scores) > 0 else np.nan,
        }
        if "pvalue" in group.columns:
            row["n_significant_0.05"] = int((group["pvalue"] < 0.05).sum())
            row["n_significant_bonferroni"] = int(
                (group["pvalue"] < 0.05 / max(1, len(group))).sum()
            )
        return pd.Series(row)

    if group_by is not None:
        return effects.groupby(group_by).apply(_stats, include_groups=False).reset_index()
    else:
        return _stats(effects).to_frame().T


def annotate_variants(
    variants: pd.DataFrame,
    annotation_bed: str,
    label_col: str = "annotation",
) -> pd.DataFrame:
    """
    Annotate variants with overlapping genomic features.

    Reads a BED file of genomic annotations (e.g., genes, regulatory
    elements) and labels each variant with overlapping features.

    Parameters
    ----------
    variants : pd.DataFrame
        Must contain ``chrom`` and ``pos`` columns.
    annotation_bed : str
        Path to a BED file (tab-separated: chrom, start, end, name).
    label_col : str
        Name of the new annotation column.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with an added annotation column.
    """
    # Load annotations
    annotations = pd.read_csv(
        annotation_bed,
        sep="\t",
        header=None,
        names=["chrom", "start", "end", "name"],
        usecols=[0, 1, 2, 3],
        comment="#",
    )

    # Build interval lookup per chromosome
    chrom_intervals = {}
    for chrom, group in annotations.groupby("chrom"):
        intervals = list(zip(group["start"], group["end"], group["name"]))
        # Sort by start for binary search
        intervals.sort(key=lambda x: x[0])
        chrom_intervals[chrom] = intervals

    # Annotate each variant
    labels = []
    for _, row in variants.iterrows():
        chrom = str(row["chrom"])
        pos = int(row["pos"]) - 1  # Convert to 0-based

        hits = []
        for c in [chrom, "chr" + chrom, chrom.replace("chr", "")]:
            if c in chrom_intervals:
                for start, end, name in chrom_intervals[c]:
                    if start <= pos < end:
                        hits.append(name)
                break

        labels.append(";".join(hits) if hits else "intergenic")

    result = variants.copy()
    result[label_col] = labels
    return result


def classify_effect_direction(
    effects: pd.DataFrame,
    score_col: str = "effect_size",
    threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Classify variants as activating, repressing, or neutral.

    Parameters
    ----------
    effects : pd.DataFrame
        Variant effect DataFrame.
    score_col : str
        Effect size column.
    threshold : float
        Minimum absolute effect to be classified as non-neutral.

    Returns
    -------
    pd.DataFrame
        With added ``direction`` column: "activating", "repressing", or "neutral".
    """
    result = effects.copy()
    scores = result[score_col].fillna(0)
    result["direction"] = np.where(
        scores > threshold, "activating",
        np.where(scores < -threshold, "repressing", "neutral"),
    )
    return result
