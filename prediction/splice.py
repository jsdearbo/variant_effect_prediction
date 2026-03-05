"""
Splice-site variant effect prediction.

Implements splice gain/loss scoring inspired by the Pangolin architecture.
Scores variants for their impact on splice donor and acceptor sites
using a dilated residual CNN. Supports both pre-trained Pangolin models
and custom splice prediction models.
"""

from typing import Optional

import numpy as np


# One-hot encoding map: N=0, A=1, C=2, G=3, T=4
_IN_MAP = np.array([
    [0, 0, 0, 0],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
])


def one_hot_encode(seq: str, strand: str = "+") -> np.ndarray:
    """
    One-hot encode a DNA sequence with strand awareness.

    For the minus strand, the sequence is reversed and complemented
    before encoding.

    Parameters
    ----------
    seq : str
        DNA sequence (A, C, G, T, N).
    strand : str
        "+" for forward, "-" for reverse complement.

    Returns
    -------
    np.ndarray
        One-hot matrix, shape (4, L).
    """
    seq = seq.upper().replace("A", "1").replace("C", "2")
    seq = seq.replace("G", "3").replace("T", "4").replace("N", "0")

    if strand == "+":
        indices = np.array(list(map(int, list(seq))))
    elif strand == "-":
        indices = np.array(list(map(int, list(seq[::-1]))))
        indices = (5 - indices) % 5  # Reverse complement
    else:
        raise ValueError(f"strand must be '+' or '-', got {strand!r}")

    return _IN_MAP[indices.astype(np.int8)].T


def compute_splice_scores(
    ref_seq: str,
    alt_seq: str,
    strand: str,
    models: list,
    distance: int = 50,
) -> dict:
    """
    Score a variant for splice site gain/loss.

    Runs both reference and alternate sequences through an ensemble
    of splice prediction models and computes per-position splice score
    differences. The worst loss and best gain across models are reported.

    Inspired by the Pangolin architecture (Zeng & Li, 2022) which uses
    4 tissue-specific output heads, each with 3 model replicates.

    Parameters
    ----------
    ref_seq : str
        Reference DNA sequence centered on the variant.
    alt_seq : str
        Alternate DNA sequence (same center, allele substituted).
    strand : str
        Gene strand ("+" or "-").
    models : list
        List of PyTorch nn.Module models. Expected to return a tensor
        of shape (1, C, L) where C includes splice probability channels.
    distance : int
        Number of positions around the variant to report scores for.

    Returns
    -------
    dict
        Keys:
        - ``loss``: np.ndarray of per-position splice loss scores (negative = loss)
        - ``gain``: np.ndarray of per-position splice gain scores (positive = gain)
        - ``max_loss``: (position_offset, score) of strongest splice loss
        - ``max_gain``: (position_offset, score) of strongest splice gain
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for splice scoring.")

    ref_encoded = one_hot_encode(ref_seq, strand)
    alt_encoded = one_hot_encode(alt_seq, strand)

    ref_tensor = torch.from_numpy(np.expand_dims(ref_encoded, axis=0)).float()
    alt_tensor = torch.from_numpy(np.expand_dims(alt_encoded, axis=0)).float()

    device = next(models[0].parameters()).device
    ref_tensor = ref_tensor.to(device)
    alt_tensor = alt_tensor.to(device)

    # Collect scores from each model head group
    # Pangolin has 4 tissue groups × 3 replicates = 12 models
    # Each model outputs 12 channels: pairs of (softmax, sigmoid)
    # Splice probability channels are at indices [1, 4, 7, 10]
    n_groups = max(1, len(models) // 3)
    all_group_diffs = []

    for g in range(n_groups):
        group_models = models[3 * g : 3 * (g + 1)]
        if not group_models:
            break

        group_scores = []
        for m in group_models:
            with torch.no_grad():
                ref_out = m(ref_tensor)[0].cpu().numpy()
                alt_out = m(alt_tensor)[0].cpu().numpy()

            # Use the sigmoid channel for this group (indices 1, 4, 7, 10)
            channel_idx = min(1 + 3 * g, ref_out.shape[0] - 1)
            ref_score = ref_out[channel_idx]
            alt_score = alt_out[channel_idx]

            if strand == "-":
                ref_score = ref_score[::-1]
                alt_score = alt_score[::-1]

            # Handle length differences from indels
            diff = alt_score - ref_score
            if len(diff) != len(ref_score):
                # Align by padding/trimming
                target_len = 2 * distance + 1
                if len(diff) > target_len:
                    start = (len(diff) - target_len) // 2
                    diff = diff[start : start + target_len]
                else:
                    diff = np.pad(diff, (0, target_len - len(diff)))

            group_scores.append(diff)

        all_group_diffs.append(np.mean(group_scores, axis=0))

    if not all_group_diffs:
        empty = np.zeros(2 * distance + 1)
        return {
            "loss": empty,
            "gain": empty,
            "max_loss": (0, 0.0),
            "max_gain": (0, 0.0),
        }

    all_group_diffs = np.array(all_group_diffs)

    # Loss: minimum across groups at each position
    loss = all_group_diffs[np.argmin(all_group_diffs, axis=0), np.arange(all_group_diffs.shape[1])]
    # Gain: maximum across groups at each position
    gain = all_group_diffs[np.argmax(all_group_diffs, axis=0), np.arange(all_group_diffs.shape[1])]

    # Window to ±distance around center
    center = len(loss) // 2
    window_start = max(0, center - distance)
    window_end = min(len(loss), center + distance + 1)

    loss_window = loss[window_start:window_end]
    gain_window = gain[window_start:window_end]

    loss_idx = int(np.argmin(loss_window))
    gain_idx = int(np.argmax(gain_window))

    return {
        "loss": loss_window,
        "gain": gain_window,
        "max_loss": (loss_idx - distance, float(loss_window[loss_idx])),
        "max_gain": (gain_idx - distance, float(gain_window[gain_idx])),
    }


def score_variant_splicing(
    chrom: str,
    pos: int,
    ref: str,
    alt: str,
    strand: str,
    models: list,
    genome_fasta: str,
    distance: int = 50,
    flank: int = 5000,
) -> dict:
    """
    End-to-end splice scoring for a single variant.

    Extracts flanking sequence from the genome, substitutes the alternate
    allele, and scores for splice gain/loss.

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
    strand : str
        Gene strand ("+" or "-").
    models : list
        Splice prediction models.
    genome_fasta : str
        Path to indexed reference FASTA.
    distance : int
        Positions around variant to score.
    flank : int
        Sequence context to extract on each side.

    Returns
    -------
    dict
        Splice scores (see ``compute_splice_scores``).
    """
    try:
        import pysam
    except ImportError:
        raise ImportError("pysam is required. Install with: pip install pysam")

    fasta = pysam.FastaFile(genome_fasta)
    try:
        start = pos - 1 - flank - distance
        end = pos + len(ref) - 1 + flank + distance
        ref_seq = fasta.fetch(chrom, max(0, start), end).upper()

        # Substitute alt allele
        offset = pos - 1 - max(0, start)
        alt_seq = ref_seq[:offset] + alt.upper() + ref_seq[offset + len(ref):]
    finally:
        fasta.close()

    return compute_splice_scores(ref_seq, alt_seq, strand, models, distance)
