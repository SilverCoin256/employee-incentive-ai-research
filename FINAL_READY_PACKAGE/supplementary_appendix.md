# Supplementary Appendix

## A. Reproducibility Artifacts

Primary pipeline:

```text
run_real_empirical_pipeline.py
```

Primary dataset:

```text
data/raw/ibm_hr_attrition.csv
```

Dataset checksum:

```text
2035074b8cc9762a3d5d94c189e02358
```

Primary outputs:

```text
results/empirical_run_summary.json
real_benchmarking_results.md
real_clustering_analysis.md
real_explainability_analysis.md
real_fairness_audit.md
real_robustness_analysis.md
```

## B. Empirical Summary

The strongest empirical finding is the predictive improvement from a salary-only benchmark to a full-feature CatBoost model:

| Model | ROC-AUC |
|---|---:|
| Salary-only CatBoost benchmark | 0.6688 |
| Full-feature CatBoost model | 0.8181 |

This supports the limited conclusion that broader behavioral and contextual signals contain predictive information beyond compensation alone in this dataset.

## C. Clustering Summary

The best clustering configuration was DBSCAN with eps = 2.0 and min_samples = 10. It produced two non-noise groups with 17.3% noise and silhouette ≈ 0.2729. These results support exploratory descriptive segmentation only.

## D. Fairness and Ethics Summary

Fairness diagnostics were conducted across gender, age band, department, compensation band, and marital status. These diagnostics identify subgroup risks but do not guarantee fairness, implement mitigation, establish compliance, or justify deployment.

## E. Known Limitation

The IBM HR Analytics dataset is synthetic and pedagogical. The appendix and manuscript therefore interpret all empirical outputs as demonstrations of the auditing pipeline mechanics rather than evidence of real workplace dynamics, causal retention mechanisms, validated behavioral types, or deployment readiness.
