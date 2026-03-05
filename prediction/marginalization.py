"""
Variant marginalization for statistical significance.

Implements the marginalization approach: for each variant, generate
shuffled background sequences, predict variant effects in each background,
and compare the real effect to the background distribution to derive
z-scores and p-values.
"""

from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd

try:
    from grelu.data.dataset import VariantMarginalizeDataset
    from grelu.utils import get_compare_func

    _HAS_GRELU = True
except ImportError:
    _HAS_GRELU = False


def marginalize_variants(
    variants: pd.DataFrame,
    model,
    genome: str = "hg38",
    seq_len: Optional[int] = None,
    devices: Union[str, int, List[int]] = "cpu",
    num_workers: int = 1,
    batch_size: int = 64,
    n_shuffles: int = 20,
    seed: Optional[int] = None,
    prediction_transform: Optional[Callable] = None,
    compare_func: Union[str, Callable] = "log2fc",
    rc: bool = False,
    max_seq_shift: int = 0,
) -> pd.DataFrame:
    """
    Test variant effects for statistical significance via marginalization.

    For each variant, creates ``n_shuffles`` dinucleotide-shuffled
    background sequences, predicts the variant effect in each, and
    computes a z-score and two-tailed p-value by comparing the real
    effect to this background distribution.

    Requires grelu for dataset construction and model inference.

    Parameters
    ----------
    variants : pd.DataFrame
        Columns: ``chrom``, ``pos``, ``ref``, ``alt``.
    model : callable
        grelu LightningModel with ``predict_on_dataset``.
    genome : str
        Reference genome name.
    seq_len : int, optional
        Input sequence length (defaults to model's training length).
    devices : str, int, or list of int
        Device(s) for inference.
    num_workers : int
        DataLoader workers.
    batch_size : int
        Batch size for inference.
    n_shuffles : int
        Number of background shuffles per variant.
    seed : int, optional
        Random seed for reproducibility.
    prediction_transform : callable, optional
        Transform applied to model output.
    compare_func : str or callable
        How to compare alt vs ref ("log2fc", "subtract", "divide").
    rc : bool
        Reverse-complement augmentation.
    max_seq_shift : int
        Sequence shift augmentation.

    Returns
    -------
    pd.DataFrame
        The input variants with added columns:
        - ``effect_size``: Real variant effect
        - ``bg_mean``: Mean effect in shuffled backgrounds
        - ``bg_std``: Std of background effects
        - ``zscore``: (effect - bg_mean) / bg_std
        - ``pvalue``: Two-tailed p-value from normal distribution
    """
    if not _HAS_GRELU:
        raise ImportError(
            "Marginalization requires grelu. "
            "Install with: pip install grelu"
        )

    import scipy.stats

    resolved_seq_len = seq_len or model.data_params["train"]["seq_len"]

    # --- Step 1: Real variant effects ---
    from prediction.effect import predict_variant_effects

    real_effects = predict_variant_effects(
        variants=variants,
        model=model,
        devices=devices,
        seq_len=resolved_seq_len,
        batch_size=batch_size,
        num_workers=num_workers,
        genome=genome,
        rc=rc,
        max_seq_shift=max_seq_shift,
        compare_func=compare_func,
        prediction_transform=prediction_transform,
    )
    variant_effects = real_effects["effect_size"].values

    # --- Step 2: Background effects ---
    ds = VariantMarginalizeDataset(
        variants=variants,
        seq_len=resolved_seq_len,
        genome=genome,
        n_shuffles=n_shuffles,
        seed=seed,
        rc=rc,
        max_seq_shift=max_seq_shift,
    )

    model.add_transform(prediction_transform)
    try:
        bg_preds = model.predict_on_dataset(
            ds,
            devices=devices,
            num_workers=num_workers,
            batch_size=batch_size,
            augment_aggfunc="mean",
        )  # Shape: (B, S, 2, ...)
    finally:
        model.reset_transform()

    # Reduce to (B, S)
    bg_preds = bg_preds.squeeze()
    if bg_preds.ndim > 2:
        # Collapse task/position dims
        while bg_preds.ndim > 3:
            bg_preds = bg_preds.mean(axis=-1)
        # bg_preds is (B, S, 2) — compare alt vs ref
        if isinstance(compare_func, str):
            compare = get_compare_func(compare_func)
            bg_effects = compare(bg_preds[:, :, 1], bg_preds[:, :, 0])
        else:
            bg_effects = compare_func(bg_preds[:, :, 1], bg_preds[:, :, 0])
    else:
        bg_effects = bg_preds

    # --- Step 3: Z-scores and p-values ---
    bg_mean = np.mean(bg_effects, axis=1)
    bg_std = np.std(bg_effects, axis=1)

    # Avoid division by zero
    bg_std_safe = np.where(bg_std > 0, bg_std, 1.0)
    zscores = (variant_effects - bg_mean) / bg_std_safe
    pvalues = scipy.stats.norm.sf(np.abs(zscores)) * 2

    result = variants.copy()
    result["effect_size"] = variant_effects
    result["bg_mean"] = bg_mean
    result["bg_std"] = bg_std
    result["zscore"] = zscores
    result["pvalue"] = pvalues

    return result


def compute_background_distribution(
    effects: np.ndarray,
    n_shuffles: int,
) -> dict:
    """
    Summarize a background effect distribution.

    Utility for custom marginalization workflows where background
    effects are computed externally.

    Parameters
    ----------
    effects : np.ndarray
        Shape: (n_variants, n_shuffles). Background effect sizes.
    n_shuffles : int
        Number of shuffles (for validation).

    Returns
    -------
    dict
        Keys: mean, std, percentile_2_5, percentile_97_5
    """
    if effects.shape[1] != n_shuffles:
        raise ValueError(
            f"Expected {n_shuffles} shuffles, got {effects.shape[1]}"
        )

    return {
        "mean": np.mean(effects, axis=1),
        "std": np.std(effects, axis=1),
        "percentile_2_5": np.percentile(effects, 2.5, axis=1),
        "percentile_97_5": np.percentile(effects, 97.5, axis=1),
    }
