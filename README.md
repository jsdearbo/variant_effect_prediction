# variant-effect-prediction

Toolkit for predicting and analyzing the functional effects of genetic variants using genomic sequence-to-function models (Borzoi, Enformer, etc.). Covers the full variant analysis pipeline: VCF parsing, filtering, ref/alt effect prediction, statistical significance testing via marginalization, splice-site scoring, and publication-quality visualization.

## Modules

### `vcf_utils/` — VCF I/O and variant processing

| Function | Description |
|---|---|
| `parse_vcf` | Read VCF/VCF.gz files into pandas DataFrames with multi-allelic splitting |
| `write_vcf` | Write variant DataFrames back to VCF format |
| `vcf_record_to_dict` | Programmatic single-variant record creation |
| `filter_variants` | Filter by allele type, insertion/deletion length, base composition, quality |
| `filter_to_snvs` | Keep only single-nucleotide variants |
| `variants_to_intervals` | Center variants in genomic windows for model input (e.g., 524,288 bp) |
| `check_reference` | Validate reference alleles against an indexed genome FASTA |
| `variant_to_ref_alt_seqs` | Extract ref and alt sequences centered on a variant |

### `prediction/` — Variant effect prediction

| Function | Description |
|---|---|
| `predict_variant_effects` | Batch prediction pipeline with ref/alt comparison (grelu or standalone) |
| `compare_predictions` | Effect size computation: log2FC, subtraction, or ratio |
| `marginalize_variants` | Statistical significance via background shuffling, z-scores, and p-values |
| `compute_background_distribution` | Summarize marginalization background statistics |
| `compute_splice_scores` | Splice gain/loss scoring with dilated CNN models (Pangolin-style) |
| `score_variant_splicing` | End-to-end splice scoring from variant coordinates |
| `one_hot_encode` | Strand-aware DNA one-hot encoding |

### `analysis/` — Post-prediction analysis and visualization

| Function | Description |
|---|---|
| `rank_variants` | Rank variants by absolute effect size |
| `summarize_effects` | Summary statistics, optionally grouped (by chromosome, gene, etc.) |
| `annotate_variants` | Overlap variants with BED annotations (genes, regulatory elements) |
| `classify_effect_direction` | Label variants as activating, repressing, or neutral |
| `plot_effect_strip` | Strip/swarm plot of effect sizes, optionally grouped |
| `plot_manhattan` | Manhattan-style genome-wide variant effect plot |
| `plot_splice_track` | Per-position splice gain/loss around a variant |

## Installation

```bash
pip install -e .                    # core only (numpy, pandas)
pip install -e ".[genome]"          # + pysam (reference validation, sequence extraction)
pip install -e ".[viz]"             # + matplotlib, seaborn
pip install -e ".[splice]"          # + PyTorch (splice prediction)
pip install -e ".[grelu]"           # + grelu (batched variant effect prediction)
pip install -e ".[all]"             # everything
```

## Quick Start

```python
from vcf_utils import parse_vcf, filter_variants, filter_to_snvs, variants_to_intervals
from prediction import predict_variant_effects, marginalize_variants
from analysis import rank_variants, plot_manhattan

# 1. Load and filter variants
variants = parse_vcf("sample.vcf.gz")
variants = filter_variants(variants, standard_bases=True, max_insert_len=0, max_del_len=0)
variants = filter_to_snvs(variants)

# 2. Predict variant effects
effects = predict_variant_effects(
    variants,
    model=model,              # grelu LightningModel or any predict_on_seqs model
    seq_len=524_288,          # Borzoi input length
    genome="hg38",
    compare_func="log2fc",
)

# 3. Statistical significance (marginalization)
significant = marginalize_variants(
    variants,
    model=model,
    genome="hg38",
    n_shuffles=20,
)

# 4. Analyze and visualize
top_hits = rank_variants(significant, top_n=50)
plot_manhattan(significant)
```

### Splice variant scoring

```python
from prediction.splice import score_variant_splicing

scores = score_variant_splicing(
    chrom="chr17", pos=43_094_580,
    ref="A", alt="G", strand="-",
    models=pangolin_models,
    genome_fasta="hg38.fa",
)
# scores["max_gain"] → (position_offset, gain_score)
# scores["max_loss"] → (position_offset, loss_score)
```

## Architecture

The prediction pipeline compares model output for reference vs. alternate allele sequences:

```
VCF → filter → center in window → extract ref/alt seqs → model(ref), model(alt) → compare → effect_size
                                                                                   ↓
                                                              marginalize (shuffle backgrounds → z-score → p-value)
```

Three independent packages with minimal coupling:
```
vcf_utils/     VCF I/O, filtering, coordinate mapping (numpy, pandas, pysam)
prediction/    Effect prediction, marginalization, splice scoring (grelu, torch)
analysis/      Summary statistics and visualization (matplotlib, seaborn)
```

## Testing

```bash
pytest tests/ -v
```

## License

MIT
