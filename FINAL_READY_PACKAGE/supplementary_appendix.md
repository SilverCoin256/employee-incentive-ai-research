# Supplementary Appendix

## A. Reproducibility Artifacts

Primary pipeline:

```text
run_real_empirical_pipeline.py
```

Primary dataset:

```text
data/ibm_hr_attrition.csv   (IBM HR Analytics Employee Attrition, synthetic, public)
```

Primary outputs (produced by one run of the pipeline):

```text
results/empirical_run_summary.json    all scalar metrics
results/arrays.npz                    raw predictions and importance arrays
figures/*.{png,pdf,svg}               regenerated from the two files above
```

All figures and tables in the manuscript are generated from these artifacts. No
metric is embedded in the plotting code.

## B. Empirical Summary

Stratified 5-fold cross-validation (seed 42). Primary model: CatBoost (400 iterations,
learning rate 0.05, depth 6, native categorical handling, positive-class weight 5:1).

| Configuration | ROC-AUC (CV) |
|---|---:|
| Full-feature CatBoost | 0.810 ± 0.019 |
| Compensation-removed (leave-group-out) | 0.793 ± 0.021 |
| Compensation-only | 0.671 ± 0.033 |

Removing the compensation feature group lowers AUC by only 0.017, indicating that
compensation is largely redundant with the remaining structural features; a
compensation-only model reaches 0.671, so non-compensation features add 0.140 AUC
over compensation alone. Logistic Regression (0.826) and Random Forest (0.815) are
statistically comparable to CatBoost in discrimination (overlapping confidence
intervals).

## C. Calibration Summary

Primary (weighted) model ECE = 0.043 (10-bin uniform, 95% CI [0.031, 0.061]); adaptive
binning 0.048. Unweighted sensitivity ECE = 0.050 (95% CI [0.039, 0.069]). The 5:1
positive-class weighting shifts mean predicted probability from 0.116 (unweighted) to
0.174 (base rate 0.161) while keeping ECE below the 0.10 governance bound under both
settings. Brier score 0.106.

## D. Clustering Summary

No clustering family reaches the 0.30 silhouette governance threshold: PCA(10)+KMeans
(k=2) 0.236, KMeans 0.170, GMM 0.122; DBSCAN produced no configuration with two or more
non-noise clusters at eps ∈ {1.5, 2.0, 2.5, 3.0}. These support a
segmentation-not-warranted finding, not operational segmentation.

## E. Fairness and Proxy Summary

Subgroup diagnostics (pooled CV, primary model) across gender, age band, marital status,
compensation band, and department report DPD, EOD, disparate-impact ratio, and Cramér's
V. The compensation band exceeds the Cramér's V escalation threshold (0.285 > 0.20).
Proxy mutual information (scikit-learn `mutual_info_classif`, KSG k-NN k=3, nats, with a
200-permutation null): MaritalStatus↔StockOptionLevel 0.426 (z≈48), AgeBand↔
TotalWorkingYears 0.386 (z≈39). These diagnostics flag subgroup and proxy risks for
review; they do not guarantee fairness, establish compliance, or justify deployment.

## F. Known Limitation

The IBM HR Analytics dataset is synthetic and pedagogical. All empirical outputs are
demonstrations of the auditing-pipeline mechanics on a controlled substrate rather than
evidence of real workplace dynamics, causal retention mechanisms, or deployment
readiness.
