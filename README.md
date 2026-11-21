# Multi-Label Toxic Comment Classification Using Machine Learning

## Project Overview

This research project investigates **multi-label toxic comment classification** using traditional machine learning and text representation techniques.

The objective is to automatically identify different forms of toxicity in online comments while examining the influence of feature representation, class imbalance, decision thresholds, and evaluation methodology on classification performance.

The project currently covers **Phase 1: Traditional Machine Learning Experiments (Cells 1–13A)**.

Deep learning, transformer models, explainability, and fairness evaluation are planned for subsequent phases.

## Dataset

**Dataset:** Jigsaw Toxic Comment Classification Challenge

**Source:** https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

The dataset contains Wikipedia comments annotated with six toxicity labels:

- `toxic`
- `severe_toxic`
- `obscene`
- `threat`
- `insult`
- `identity_hate`

This is a multi-label classification problem, meaning that an individual comment may belong to multiple toxicity categories simultaneously.

### Dataset Statistics

| Dataset Partition | Samples |
|---|---:|
| Original training dataset | 159,571 |
| Model training split | 127,656 |
| Validation split | 31,915 |
| Original threshold-tuning subset | 15,957 |
| Original audit subset | 15,958 |
| Official test examples with valid labels | 63,978 |

An iterative multi-label stratified split was used to approximately preserve label proportions.

## Methodology

The experimental workflow includes:

1. Dataset loading and exploratory data analysis.
2. Text preprocessing and multi-label stratification.
3. Word-level TF-IDF feature extraction.
4. Training and comparison of traditional machine learning models.
5. Character-level TF-IDF experiments.
6. Combined word and character TF-IDF representations.
7. Per-label decision-threshold optimization.
8. Class-weighted training.
9. Normalized duplicate-leakage detection and sensitivity analysis.

### Machine Learning Algorithms

The following algorithms were evaluated:

- Logistic Regression
- Multinomial Naive Bayes
- Linear Support Vector Machine (SVM)
- Decision Tree
- Random Forest

### Feature Extraction

**Word TF-IDF:** Word unigrams and bigrams, with up to 100,000 features.

**Character TF-IDF:** Character 3–5-grams, with up to 150,000 features.

**Combined TF-IDF:** Concatenated word and character representations, containing up to 250,000 features.

The experiments investigate whether character-level information and combined representations improve classification performance.

## Experimental Results

### Traditional Machine Learning Baselines

Results on the original full validation set, before per-label threshold optimization:

| Model | Macro F1 |
|---|---:|
| Linear SVM — Word TF-IDF | 0.5629 |
| Logistic Regression — Word TF-IDF | 0.4663 |
| Decision Tree — Word TF-IDF | 0.3406 |
| Naive Bayes — Word TF-IDF | 0.2573 |
| Random Forest — Word TF-IDF | 0.0042 |

Random Forest demonstrated useful ranking performance despite its low F1 at the default threshold.

### Feature Representation Comparison

| Linear SVM Representation | Macro F1 |
|---|---:|
| Word TF-IDF | 0.5629 |
| Character TF-IDF | 0.5835 |
| Combined Word + Character TF-IDF | **0.5839** |

Character features improved performance compared with the word-only SVM configuration.

The combined representation achieved a slightly higher overall Macro F1, although its advantage over character-only features was very small.

### Threshold Optimization

Separate decision thresholds were optimized for each toxicity label using a dedicated validation-tuning subset.

On the original validation audit subset:

| Model | Default Macro F1 | Tuned Macro F1 |
|---|---:|---:|
| Combined SVM | 0.5668 | **0.6251** |
| Character SVM | 0.5744 | 0.6220 |
| Character Logistic Regression | 0.4664 | 0.6091 |
| Word SVM | 0.5466 | 0.6082 |
| Random Forest | 0.0032 | 0.5502 |

These results demonstrate that the decision threshold can substantially affect classification performance, especially for imbalanced labels.

### Class-Weighting Experiments

Class-weighted Logistic Regression and Linear SVM were evaluated.

On the original validation audit subset:

| Model | Tuned Macro F1 |
|---|---:|
| Unweighted Word Logistic Regression | 0.5770 |
| Weighted Word Logistic Regression | 0.6009 |
| Unweighted Combined SVM | **0.6251** |
| Weighted Combined SVM | 0.6219 |

Class weighting improved Logistic Regression but did not improve the tuned Macro F1 of the combined SVM.

## Data Leakage Audit and Sensitivity Analysis

A normalized-text duplicate audit was conducted to investigate potential overlap between training and validation comments.

### Findings

- 71 normalized duplicate-text groups appeared in both training and validation.
- 80 training rows belonged to these overlapping groups.
- 77 validation rows were affected.
- Removing the 77 affected validation rows left 31,838 validation examples.

Threshold tuning and audit subsets were also cleaned to remove relevant normalized-text overlap.

### Corrected Evaluation

| Model | Original Full-Validation Macro F1 | Clean Full-Validation Macro F1 |
|---|---:|---:|
| Combined SVM | 0.5839 | 0.5841 |
| Character SVM | 0.5835 | 0.5836 |
| Word SVM | 0.5629 | 0.5628 |

For the Combined SVM, the cleaned audit subset contained **15,923 comments**.

| Evaluation | Macro F1 |
|---|---:|
| Default decision threshold | 0.5675 |
| Optimized label-specific thresholds | **0.6258** |

The cleaned evaluation showed that the identified duplicate overlap had minimal impact on the reported exploratory findings.

These results remain exploratory because the original validation data had previously been used during model comparison. A duplicate-safe evaluation protocol and independent confirmatory assessment are recommended before making definitive generalization claims.

## Key Research Findings

1. Character-level TF-IDF features can improve traditional toxic-comment classification.
2. Combining word and character features provides useful lexical representations, although the additional improvement over character-only SVM was small.
3. Label-specific threshold optimization substantially improves F1 in several model configurations.
4. Class weighting has model-dependent effects.
5. Rare toxicity labels require careful evaluation because of limited positive examples.
6. Leakage checks and consistent evaluation protocols are essential for reliable research.

## Technologies Used

- Python
- Google Colab
- Pandas
- NumPy
- Scikit-learn
- SciPy
- Matplotlib
- Joblib
- Iterative Stratification

## Project Status

**Phase 1 — Completed:** Traditional ML baselines, TF-IDF comparisons, threshold optimization, class weighting, and duplicate-leakage sensitivity analysis.

**Future Work:** Deep learning and transformer comparisons, explainability using LIME/SHAP, identity-related fairness auditing, statistical uncertainty analysis, and confirmatory evaluation.

## How to Use

The executed experiments are contained in the Google Colab notebook `multi_label.ipynb`.

The Jigsaw dataset should be downloaded from the official Kaggle competition. In the original Colab environment, the dataset archive is stored at:

`/content/jigsaw-toxic-comment-classification-challenge.zip`

The notebook documents the experimental workflow and corresponding outputs.

## Research References

1. Davidson et al. (2017), *Automated Hate Speech Detection and the Problem of Offensive Language*.
2. van Aken et al. (2018), *Challenges for Toxic Comment Classification: An In-Depth Error Analysis*.
3. Jigsaw Toxic Comment Classification Challenge, Kaggle.

The reference papers inform the research design; the experiments reported in this repository are not claimed to be exact reproductions of those studies.

---

**Note:** This repository documents ongoing academic research. Reported Phase 1 validation results should not be interpreted as final independent test performance.
