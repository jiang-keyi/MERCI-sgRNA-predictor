# MERCI

**Multi-task residual learning for sgRNA efficiency prediction across Cas9 variants**

MERCI jointly predicts the on-target editing efficiency of WT-SpCas9, eSpCas9(1.1)
and SpCas9-HF1 sgRNAs with an interpretable tree-ensemble framework that combines:

1. **Uncertainty-aware sample weighting** (Module A) — per-sgRNA weights derived from
   ensemble prediction disagreement, down-weighting low-confidence guides without
   discarding data.
2. **Feature-subspace bagging** (Module B) — 20 LightGBM base learners trained on
   complementary random 70% feature subsets.
3. **Multi-task residual correction** (Module C) — a shared WT-SpCas9 task plus two
   residual tasks (ΔeSp, ΔHF1) capturing variant-specific offsets.
4. **Stacking meta-learner** (CatBoost) — integrates the base predictions.

The paper reports internal validation Spearman correlations of 0.81/0.80/0.80
(5-seed mean) on the three variants, a mean Spearman of 0.368 across ten
independent external test sets (16,138 sgRNAs after overlap filtering), and
three-way same-test-set comparison against DeepHF and DeepMEns.

## Repository layout

```
code/                 model implementation (LightGBM/CatBoost pipelines)
scripts/              reproducibility scripts for the paper revision
  ablation_multiseed.py          multi-seed ablation (7 configs × 5 seeds)
  feature_family_ablation.py     feature-family ablation
  benchmark_deephf.py            DeepHF same-test-set benchmark
  benchmark_deepmeans.py         DeepMEns same-test-set benchmark
  make_fig1_architecture.py      Figure 1 (architecture)
  make_fig3_ablation.py          Figure 3 (multi-seed ablation)
  make_pandoc_outputs.py         paper -> docx/html conversion
  fill_paper_numbers.py          results -> manuscript number filling
data/processed/       merged training table + 225-dimensional feature table
results/              all result files (JSON/CSV) from the revision experiments
figures/              all manuscript figures
```

## Requirements

- Python 3.9+; `lightgbm`, `catboost`, `numpy`, `pandas`, `scikit-learn`,
  `scipy`, `matplotlib`
- TensorFlow 2.10 (CPU is sufficient) is required only for running the DeepHF
  baseline weights; MERCI itself is a pure gradient-boosting pipeline.

## Reproducing the main results

```bash
python scripts/ablation_multiseed.py        # multi-seed ablation -> ablation_multiseed.json/.csv
python scripts/feature_family_ablation.py   # feature-family ablation
python scripts/benchmark_deephf.py          # DeepHF external benchmark (needs DeepHF weights)
python scripts/benchmark_deepmeans.py       # DeepMEns external benchmark
python scripts/make_fig3_ablation.py        # regenerate Figure 3
```

Processed training/validation tables are included under `data/processed/`.
Raw screening datasets are from the public DeepHF repository (Wang et al. 2019)
and the original publications of each external test set.

## License

MIT — see LICENSE.
