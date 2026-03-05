"""
Variant effect prediction.

Core prediction pipeline that scores variants by comparing model
predictions for reference vs. alternate allele sequences. Supports
batch processing, reverse-complement augmentation, and configurable
comparison functions (log2 fold-change, subtraction, etc.).
"""

from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

try:
    from grelu.data.dataset import VariantDataset
    from grelu.utils import get_compare_func

    _HAS_GRELU = True
except ImportError:
    _HAS_GRELU = False


def compare_predictions(
    ref_preds: np.ndarray,
    alt_preds: np.ndarray,
    method: str = "log2fc",
) -> np.ndarray:
    """
    Compare reference and alternate predictions.

    Parameters
    ----------
    ref_preds : np.ndarray
        Predictions for the reference allele. Shape: (N,) or (N, T).
    alt_preds : np.ndarray
        Predictions for the alternate allele. Same shape as ref_preds.
    method : str
        Comparison method:
        - ``"log2fc"``: log2(alt / ref), with clipping to avoid log(0)
        - ``"subtract"``: alt - ref
        - ``"divide"``: alt / ref

    Returns
    -------
    np.ndarray
        Effect sizes, same shape as inputs.
    """
    if method == "log2fc":
        ratio = np.clip(alt_preds, 1e-8, None) / np.clip(ref_preds, 1e-8, None)
        return np.log2(ratio)
    elif method == "subtract":
        return alt_preds - ref_preds
    elif method == "divide":
        return alt_preds / np.clip(ref_preds, 1e-8, None)
    else:
        raise ValueError(f"Unknown comparison method: {method!r}")


def predict_variant_effects(
    variants: pd.DataFrame,
    model,
    seq_len: Optional[int] = None,
    genome: str = "hg38",
    batch_size: int = 64,
    num_workers: int = 1,
    devices: Union[int, str] = "cpu",
    rc: bool = False,
    max_seq_shift: int = 0,
    compare_func: Optional[Union[str, Callable]] = "log2fc",
    prediction_transform: Optional[Callable] = None,
) -> pd.DataFrame:
    """
    Predict functional effects of genetic variants.

    For each variant, extracts reference and alternate sequences centered
    on the variant, runs model inference on both, and computes an effect
    size by comparing predictions.

    When grelu is available, uses ``VariantDataset`` for efficient batched
    inference with optional reverse-complement augmentation and sequence
    shifting. Falls back to a simple per-variant loop otherwise.

    Parameters
    ----------
    variants : pd.DataFrame
        Columns: ``chrom``, ``pos``, ``ref``, ``alt``.
    model : callable
        A model with a ``predict_on_dataset`` method (grelu LightningModel)
        or a ``predict_on_seqs(seqs, device=...)`` method.
    seq_len : int, optional
        Input sequence length. Defaults to model's training seq_len.
    genome : str
        Reference genome name (e.g. "hg38", "hg19").
    batch_size : int
        Batch size for inference.
    num_workers : int
        DataLoader workers.
    devices : int or str
        Device(s) for inference.
    rc : bool
        Average predictions over forward and reverse complement.
    max_seq_shift : int
        Maximum bases to shift sequences for augmentation.
    compare_func : str, callable, or None
        How to compare alt vs ref predictions:
        "log2fc", "subtract", "divide", or a custom callable(alt, ref).
        If None, returns raw ref and alt predictions.
    prediction_transform : callable, optional
        Transform to apply to raw model output before comparison.

    Returns
    -------
    pd.DataFrame
        The input variants DataFrame with added columns:
        - ``effect_size``: Scalar effect for each variant
        - ``ref_pred``, ``alt_pred``: Raw predictions (if compare_func is None)
    """
    if _HAS_GRELU and hasattr(model, "predict_on_dataset"):
        return _predict_with_grelu(
            variants, model, seq_len, genome, batch_size, num_workers,
            devices, rc, max_seq_shift, compare_func, prediction_transform,
        )
    else:
        return _predict_simple(
            variants, model, seq_len, compare_func, prediction_transform, devices,
        )


def _predict_with_grelu(
    variants, model, seq_len, genome, batch_size, num_workers,
    devices, rc, max_seq_shift, compare_func, prediction_transform,
):
    """Predict using grelu's VariantDataset for batched inference."""
    resolved_seq_len = seq_len or model.data_params["train"]["seq_len"]

    dataset = VariantDataset(
        variants,
        seq_len=resolved_seq_len,
        genome=genome,
        rc=rc,
        max_seq_shift=max_seq_shift,
    )

    model.add_transform(prediction_transform)
    try:
        preds = model.predict_on_dataset(
            dataset,
            devices=devices,
            num_workers=num_workers,
            batch_size=batch_size,
            augment_aggfunc="mean",
        )  # Shape: (B, 2, T, L)
    finally:
        model.reset_transform()

    result = variants.copy()

    if compare_func is not None:
        if isinstance(compare_func, str):
            grelu_func = get_compare_func(compare_func)
            effects = grelu_func(preds[:, 1, :, :], preds[:, 0, :, :])
        else:
            effects = compare_func(preds[:, 1, :, :], preds[:, 0, :, :])

        # Reduce to scalar per variant
        while effects.ndim > 1:
            effects = effects.mean(axis=-1)
        result["effect_size"] = effects
    else:
        ref_preds = preds[:, 0, :, :]
        alt_preds = preds[:, 1, :, :]
        while ref_preds.ndim > 1:
            ref_preds = ref_preds.mean(axis=-1)
            alt_preds = alt_preds.mean(axis=-1)
        result["ref_pred"] = ref_preds
        result["alt_pred"] = alt_preds

    return result


def _predict_simple(
    variants, model, seq_len, compare_func, prediction_transform, devices,
):
    """Fallback prediction loop without grelu VariantDataset."""
    from vcf_utils.coordinates import variant_to_ref_alt_seqs

    device = devices if isinstance(devices, str) else "cpu"

    if seq_len is None:
        if hasattr(model, "data_params"):
            seq_len = model.data_params["train"]["seq_len"]
        else:
            raise ValueError("seq_len must be specified when model has no data_params")

    ref_seqs = []
    alt_seqs = []
    valid_mask = []

    for _, row in variants.iterrows():
        try:
            ref_s, alt_s = variant_to_ref_alt_seqs(
                row["chrom"], row["pos"], row["ref"], row["alt"],
                genome_fasta="",  # Requires genome_fasta in caller
                seq_len=seq_len,
            )
            ref_seqs.append(ref_s)
            alt_seqs.append(alt_s)
            valid_mask.append(True)
        except Exception:
            ref_seqs.append(None)
            alt_seqs.append(None)
            valid_mask.append(False)

    # Batch predict valid variants
    valid_refs = [s for s, m in zip(ref_seqs, valid_mask) if m]
    valid_alts = [s for s, m in zip(alt_seqs, valid_mask) if m]

    ref_preds = _batch_predict(model, valid_refs, prediction_transform, device)
    alt_preds = _batch_predict(model, valid_alts, prediction_transform, device)

    result = variants.copy()
    result["effect_size"] = np.nan

    if compare_func is not None:
        if isinstance(compare_func, str):
            effects = compare_predictions(ref_preds, alt_preds, method=compare_func)
        else:
            effects = compare_func(alt_preds, ref_preds)

        valid_indices = [i for i, m in enumerate(valid_mask) if m]
        for idx, effect in zip(valid_indices, effects):
            result.at[result.index[idx], "effect_size"] = float(effect)
    else:
        result["ref_pred"] = np.nan
        result["alt_pred"] = np.nan
        valid_indices = [i for i, m in enumerate(valid_mask) if m]
        for idx, rp, ap in zip(valid_indices, ref_preds, alt_preds):
            result.at[result.index[idx], "ref_pred"] = float(rp)
            result.at[result.index[idx], "alt_pred"] = float(ap)

    return result


def _batch_predict(model, seqs, prediction_transform, device, batch_size=32):
    """Run model prediction on a list of sequences."""
    all_preds = []
    for i in range(0, len(seqs), batch_size):
        batch = seqs[i : i + batch_size]
        preds = model.predict_on_seqs(batch, device=device)
        if prediction_transform is not None:
            preds = prediction_transform(preds)
        all_preds.append(preds)

    if not all_preds:
        return np.array([])

    combined = np.concatenate(all_preds, axis=0)
    # Reduce to scalar per sequence
    while combined.ndim > 1:
        combined = combined.mean(axis=-1)
    return combined
