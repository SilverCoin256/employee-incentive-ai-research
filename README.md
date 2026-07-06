# employee-incentive-ai-research

Code and evaluation pipeline for:

**"A Governance Aware Framework for Auditing Calibration Explainability and Fairness in HR Attrition Prediction Models"**
*Discover Artificial Intelligence (Springer Nature)*

---

## What this is

A single reproducible pipeline that audits an HR attrition model across calibration,
uncertainty, explainability, and fairness. The contribution is the evaluation
architecture, not a new model. The IBM HR Analytics dataset is synthetic and public,
chosen so the whole pipeline can be replicated without data-access barriers.

Everything reported in the manuscript is produced by one run of **`pipeline.py`**
(seed = 42). It writes `results/results.json` and regenerates every figure
(`figures/`) directly from the run's raw predictions — no values are embedded in
the plotting code. Run it with:

```
pip install -r requirements.txt        # numpy, pandas, scikit-learn, catboost, shap, matplotlib, scipy
python3 pipeline.py                     # -> results/results.json + figures/*.{png,pdf,svg}
```

Headline outputs (CatBoost, 5-fold CV, protected attributes excluded, native
categoricals, 5:1 positive-class weight): ROC-AUC 0.815 (σ 0.034), ECE 0.032,
Brier 0.102; leave-group-out compensation drop 0.022, compensation-only 0.680;
age-band EOD 0.353, compensation-band Cramér's V 0.281; proxy MI
StockOptionLevel→MaritalStatus 0.426, TotalWorkingYears→Age 0.582.

The earlier `run_real_empirical_pipeline.py` / `generate_figures.py` scripts are
retained under `legacy/` for provenance; `pipeline.py` is the single authoritative
reproducer for the current manuscript.

---

## Repository structure

```
run_real_empirical_pipeline.py   main pipeline: 5-fold CV (4 model families),
                                  leave-group-out compensation ablation,
                                  calibration (ECE + bootstrap CI, adaptive binning),
                                  bootstrap AUC CI, 7-seed sweep, SHAP + permutation
                                  with faithfulness/sanity checks, DPD/EOD/DI/Cramér's V
                                  fairness, proxy-MI audit with permutation null,
                                  DBSCAN/KMeans/PCA-KMeans/GMM clustering
generate_figures.py               figures, drawn strictly from results/
results/empirical_run_summary.json  all scalar metrics from the run
results/arrays.npz                  raw predictions + importance arrays for figures
requirements.txt                  pinned dependencies
data/README.md                    how to obtain the dataset
figures/                          generated outputs (run the script to regenerate)
```

Preprocessing is leakage-controlled: encoding/scaling/imputation are fit inside each
training fold (scikit-learn `Pipeline`/`ColumnTransformer`); CatBoost uses native
categorical handling. Protected attributes (gender, age, marital status) are excluded
from the predictor set and used only for fairness auditing.

---

## Running

```bash
pip install -r requirements.txt
# place the dataset at data/ibm_hr_attrition.csv (see data/README.md)
python run_real_empirical_pipeline.py     # writes results/
python generate_figures.py                # writes figures/ from results/
```

`shap` is optional; if it is not installed the pipeline uses CatBoost's native
(exact TreeSHAP) Shapley values.

---

## Key results

From `results/empirical_run_summary.json` — stratified 5-fold CV, seed 42,
primary model = CatBoost (400 iters, lr 0.05, depth 6, native cats, positive-class
weight 5:1):

| Metric | Value |
|--------|-------|
| ROC-AUC (5-fold CV, primary) | 0.810 ± 0.019 (95% CI [0.793, 0.827]) |
| ROC-AUC (holdout / bootstrap CI) | 0.783 / [0.695, 0.854] |
| ROC-AUC (7-seed range, mean) | 0.773–0.836 (0.805) |
| Brier score | 0.106 |
| ECE (10-bin, primary) | 0.043 (95% CI [0.031, 0.061]); adaptive 0.048 |
| ECE (unweighted sensitivity) | 0.050 (95% CI [0.039, 0.069]) |
| Model comparison (CV AUC) | LogReg 0.826, RF 0.815, CatBoost 0.810, Dummy 0.500 |
| Ablation | full 0.810; comp-removed 0.793 (Δ 0.017); comp-only 0.671 |
| Proxy MI (nats) | MaritalStatus↔StockOptionLevel 0.426 (z≈48); AgeBand↔TotalWorkingYears 0.386 (z≈39) |
| Fairness escalation (Cramér's V > 0.20) | compensation band 0.285 |

Clustering: no family (DBSCAN, KMeans, PCA+KMeans, GMM) reaches the 0.30 silhouette
threshold (max 0.236), so the pipeline returns a segmentation-not-warranted finding.

Model AUCs overlap in their confidence intervals; CatBoost is used as the primary
model for its calibration behaviour and native categorical handling, not a claim of
superior discrimination.

---

## Limitations

The IBM HR dataset is synthetic and not from a real organization. All results
demonstrate evaluation-framework mechanics; generalization to real workforces requires
external validation with appropriate data provenance. See the manuscript's Scope
Conditions.

---

## Environment

Tested on Python 3.11. Core dependencies: catboost, scikit-learn, numpy, pandas, scipy,
matplotlib. `shap` optional.

## Contact

Shaurya Gupta — shauryagupta042@gmail.com
