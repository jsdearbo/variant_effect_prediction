"""
Variant effect visualization.

Publication-quality plots for variant effect analysis: Manhattan-style
plots, effect size distributions, and splice score tracks.
"""

from typing import Optional

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

try:
    import seaborn as sns

    _HAS_SNS = True
except ImportError:
    _HAS_SNS = False


def _require_matplotlib():
    if not _HAS_MPL:
        raise ImportError("matplotlib is required for plotting. pip install matplotlib")


def plot_effect_strip(
    effects: pd.DataFrame,
    score_col: str = "effect_size",
    group_col: Optional[str] = None,
    threshold: Optional[float] = None,
    figsize: tuple = (10, 5),
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Strip plot of variant effect sizes.

    Each point is a variant, optionally grouped by a categorical variable
    (e.g., chromosome, gene, functional class). Optionally draws a
    significance threshold line.

    Parameters
    ----------
    effects : pd.DataFrame
        Must contain ``score_col``.
    score_col : str
        Column with effect sizes.
    group_col : str, optional
        Column to group variants on the x-axis.
    threshold : float, optional
        Draw horizontal lines at ±threshold.
    figsize : tuple
        Figure size if creating new figure.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.

    Returns
    -------
    matplotlib.axes.Axes
    """
    _require_matplotlib()

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    df = effects.dropna(subset=[score_col]).copy()

    if group_col and _HAS_SNS:
        sns.stripplot(
            data=df, x=group_col, y=score_col,
            alpha=0.6, size=4, jitter=True, ax=ax,
        )
        ax.set_xlabel(group_col)
    else:
        ax.scatter(range(len(df)), df[score_col], alpha=0.6, s=15, c="steelblue")
        ax.set_xlabel("Variant index")

    ax.set_ylabel(score_col.replace("_", " ").title())
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    if threshold is not None:
        ax.axhline(threshold, color="red", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.axhline(-threshold, color="red", linewidth=0.8, linestyle="--", alpha=0.7)

    ax.set_title("Variant Effect Sizes")
    return ax


def plot_manhattan(
    effects: pd.DataFrame,
    score_col: str = "effect_size",
    chrom_col: str = "chrom",
    pos_col: str = "pos",
    pvalue_col: Optional[str] = "pvalue",
    significance_threshold: float = 5e-8,
    figsize: tuple = (14, 5),
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Manhattan-style plot of variant effects across the genome.

    The y-axis shows -log10(p-value) if available, otherwise absolute
    effect size. Chromosomes are color-alternated.

    Parameters
    ----------
    effects : pd.DataFrame
        Must contain ``chrom_col`` and ``pos_col``.
    score_col : str
        Effect size column (used if pvalue_col not in DataFrame).
    chrom_col : str
        Chromosome column.
    pos_col : str
        Position column.
    pvalue_col : str, optional
        P-value column for -log10 transform.
    significance_threshold : float
        Genome-wide significance line.
    figsize : tuple
        Figure size.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.

    Returns
    -------
    matplotlib.axes.Axes
    """
    _require_matplotlib()

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    df = effects.copy()

    # Determine y-axis values
    if pvalue_col and pvalue_col in df.columns:
        pvals = pd.to_numeric(df[pvalue_col], errors="coerce")
        df["_y"] = -np.log10(np.clip(pvals, 1e-300, 1))
        ylabel = "$-\\log_{10}(p)$"
        threshold_y = -np.log10(significance_threshold)
    else:
        df["_y"] = df[score_col].abs()
        ylabel = f"|{score_col}|"
        threshold_y = None

    df = df.dropna(subset=["_y"])

    # Sort chromosomes
    chrom_order = _sort_chromosomes(df[chrom_col].unique())
    df["_chrom_idx"] = df[chrom_col].map({c: i for i, c in enumerate(chrom_order)})
    df = df.sort_values(["_chrom_idx", pos_col])

    # Compute cumulative positions
    chrom_offsets = {}
    offset = 0
    for chrom in chrom_order:
        chrom_offsets[chrom] = offset
        chrom_df = df[df[chrom_col] == chrom]
        if len(chrom_df) > 0:
            offset += chrom_df[pos_col].max() + 1_000_000

    df["_x"] = df.apply(lambda r: chrom_offsets.get(r[chrom_col], 0) + r[pos_col], axis=1)

    # Alternate chromosome colors
    colors = []
    palette = ["#1f77b4", "#aec7e8"]
    for _, row in df.iterrows():
        cidx = chrom_order.index(row[chrom_col]) if row[chrom_col] in chrom_order else 0
        colors.append(palette[cidx % 2])

    ax.scatter(df["_x"], df["_y"], c=colors, s=8, alpha=0.7, edgecolors="none")

    # Chromosome labels
    chrom_centers = {}
    for chrom in chrom_order:
        chrom_df = df[df[chrom_col] == chrom]
        if len(chrom_df) > 0:
            chrom_centers[chrom] = chrom_df["_x"].median()

    ax.set_xticks(list(chrom_centers.values()))
    ax.set_xticklabels(
        [c.replace("chr", "") for c in chrom_centers.keys()],
        rotation=45, fontsize=8,
    )

    # Significance line
    if threshold_y is not None:
        ax.axhline(threshold_y, color="red", linewidth=0.8, linestyle="--", alpha=0.7)

    ax.set_xlabel("Chromosome")
    ax.set_ylabel(ylabel)
    ax.set_title("Variant Effect Manhattan Plot")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return ax


def plot_splice_track(
    splice_scores: dict,
    distance: int = 50,
    figsize: tuple = (12, 4),
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot splice gain/loss scores around a variant.

    Shows per-position splice score changes, with gain in blue and
    loss in red. The variant position is marked at center.

    Parameters
    ----------
    splice_scores : dict
        Output of ``compute_splice_scores``:
        ``loss`` and ``gain`` arrays, plus ``max_loss`` and ``max_gain``.
    distance : int
        Number of positions plotted on each side of variant.
    figsize : tuple
        Figure size.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.

    Returns
    -------
    matplotlib.axes.Axes
    """
    _require_matplotlib()

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    loss = splice_scores["loss"]
    gain = splice_scores["gain"]
    positions = np.arange(-distance, -distance + len(gain))

    # Gain (positive) in blue, Loss (negative) in red
    ax.fill_between(positions, 0, gain, where=(gain > 0), color="#4393c3", alpha=0.7, label="Splice gain")
    ax.fill_between(positions, 0, loss, where=(loss < 0), color="#d6604d", alpha=0.7, label="Splice loss")

    # Overlay as line
    ax.plot(positions, gain, color="#2166ac", linewidth=0.8)
    ax.plot(positions, loss, color="#b2182b", linewidth=0.8)

    # Mark variant position
    ax.axvline(0, color="black", linewidth=1, linestyle=":", alpha=0.5, label="Variant")

    # Annotate peak scores
    max_gain_pos, max_gain_val = splice_scores["max_gain"]
    max_loss_pos, max_loss_val = splice_scores["max_loss"]

    if abs(max_gain_val) > 0.01:
        ax.annotate(
            f"+{max_gain_val:.2f}",
            xy=(max_gain_pos, max_gain_val),
            fontsize=8, color="#2166ac",
            ha="center", va="bottom",
        )
    if abs(max_loss_val) > 0.01:
        ax.annotate(
            f"{max_loss_val:.2f}",
            xy=(max_loss_pos, max_loss_val),
            fontsize=8, color="#b2182b",
            ha="center", va="top",
        )

    ax.set_xlabel("Position relative to variant (bp)")
    ax.set_ylabel("Splice score change")
    ax.set_title("Splice Site Effect")
    ax.legend(fontsize=8, loc="upper right")
    ax.axhline(0, color="gray", linewidth=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return ax


def _sort_chromosomes(chroms):
    """Sort chromosome names naturally (chr1, chr2, ..., chrX, chrY)."""
    def _key(c):
        c_clean = c.replace("chr", "")
        try:
            return (0, int(c_clean))
        except ValueError:
            return (1, c_clean)
    return sorted(chroms, key=_key)
