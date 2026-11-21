# Phase 1 Addendum: Normalized-Text Duplicate Leakage Sensitivity

This appendix supplements `research_phase1.pdf` and `research_phase1.pptx`, which were created **before** Cell 13A. The source of the numbers below is the executed Colab output supplied by the researcher. They were **not independently reproduced** in this GitHub starter package.

## Diagnostic

Text normalization: lowercasing, collapsing repeated whitespace, and trimming whitespace. There were **71 normalized duplicate groups shared between training and validation** (80 training rows, 77 validation rows). Training size: 127,656; original validation: 31,915; validation after excluding the 77 overlapping rows: **31,838**.

The original threshold tuning subset of 15,957 became **15,911** after cleaning. The original audit subset of 15,958 became **15,923** after removing training-overlap examples and 4 audit rows that overlapped with tuning after normalization. Neither the models nor the TF-IDF vectorizers were retrained.

## Main findings

- Combined Word+Character Linear SVM, **clean audit**, default threshold: Macro F1 **0.5675**.
- Combined Word+Character Linear SVM, **clean audit**, per-label tuned thresholds: Macro F1 **0.6258**; same-sample difference **+0.0582**.
- Combined SVM full-validation fixed-threshold Macro F1 changed from **0.5839** to **0.5841** after excluding train-overlap examples.
- Model rankings and conclusions were largely unchanged in this sensitivity analysis.
- This is a *sensitivity analysis*, not leakage-free retraining: models previously saw some overlapping train/validation textual content, and model configurations had already been selected using validation records. The audit also had been used for model selection, so it does not qualify as an untouched confirmation set.
- The 47 positive `threat` examples in the clean audit make rare-class F1 estimates variable. Independent group-aware evaluation and uncertainty intervals remain recommended before final claims.

## Data files

- `results/leakage_sensitivity_results.csv` — default-threshold full-validation before vs after exclusion.
- `results/clean_audit_comparison.csv` — default vs tuned F1 and AP on cleaned audit subsets.
- `results/clean_combined_svm_per_class.csv` — six-label metrics and thresholds on clean audit.

## Reproducibility note

The CSV files in this starter were transcribed from the provided Colab results. After downloading the original Colab notebook, include Cell 13A and its outputs. For better auditability, attach generated outputs from that notebook and, if appropriate, non-sensitive per-example prediction arrays separately (do not include raw text or Kaggle dataset files in a public repo).
