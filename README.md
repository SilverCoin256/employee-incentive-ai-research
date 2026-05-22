# employee-incentive-ai-research

Code and evaluation pipeline for:

**"Calibration, Explainability, and Fairness Auditing in HR Attrition Prediction: A Governance-Aware Evaluation Framework for Enterprise AI under Constrained Data Conditions"**  
*Discover Artificial Intelligence (Springer Nature)*

---

## What this is

This paper evaluates a CatBoost-based attrition prediction model on the IBM HR Analytics dataset using a nine-stage governance-instrumented framework. The goal is not a new model — it's a structured evaluation architecture that produces calibration diagnostics, fairness escalation signals, and interpretability outputs with explicit interpretive boundaries.

The main contribution is the evaluation design itself. The IBM dataset is synthetic and public, selected deliberately so anyone can replicate the full pipeline without data access barriers.

---

## Repository structure

```
hardened_pipeline.py     main evaluation pipeline (5-fold CV, fairness, proxy audit, clustering)
generate_figures.py      figure generation for the manuscript (all 12 figures)
hardened_results.json    verified output snapshot from the pipeline
requirements.txt         pinned Python dependencies
data/README.md           how to get the dataset
figures/                 generated figure outputs (not tracked — run the script to regenerate)
```

---

## Running the pipeline

### 1. Install dependencies

Python 3.11+ recommended.

```bash
pip install -r requirements.txt
```

### 2. Get the dataset

Download the IBM HR Analytics Employee Attrition dataset from Kaggle and place it at `data/ibm_hr_attrition.csv`:

https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

See `data/README.md` for details.

### 3. Run the evaluation

```bash
python hardened_pipeline.py
```

Output is written to `hardened_results.json`. The terminal also prints all fold-level metrics and a LaTeX-ready summary.

### 4. Regenerate figures

```bash
python generate_figures.py
```

Figures are saved to `figures/` as SVG, PNG (600 dpi), and PDF.

---

## Key results

From `hardened_results.json` — 5-fold stratified CV, random_state=42:

| Metric | Value |
|--------|-------|
| AUC | 0.8014 ± 0.0248 |
| Brier score | 0.109 ± 0.009 |
| ECE (pooled, 10-bin) | 0.053 |
| DPD (Gender) | 0.015 |
| EOD (Gender) | 0.121 |
| Proxy MI — MaritalStatus top feature | 0.426 (StockOptionLevel) |

Clustering: all four families (DBSCAN, KMeans, PCA+KMeans, GMM) fell below the 0.30 silhouette governance threshold. This is a diagnostic outcome, not something omitted — the paper explicitly reports it as a negative segmentation finding.

---

## Limitations

The IBM HR dataset is synthetic and not from a real organization. All results demonstrate evaluation framework mechanics. Generalization to real workforces requires external validation with appropriate data provenance. The paper's Scope Conditions section lays this out explicitly.

The `FINAL_READY_PACKAGE/` folder contains an earlier draft PDF from a prior submission round — kept here for reference but is not the current manuscript.

---

## Environment

Tested on Python 3.11, macOS. Should work on Linux without changes.

Core dependencies: catboost 1.2.7, scikit-learn 1.5.2, numpy, pandas, scipy, shap, matplotlib, seaborn.

---

## Contact

Shaurya Gupta — shauryagupta042@gmail.com
