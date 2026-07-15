# Numerical traceability map

Every value reported in the manuscript traces to a named key in
`results/results.json`, produced by a single run of `pipeline.py` (seed 42).
Paths below use dot notation into that JSON file. Rounding in the manuscript is
to three decimals unless noted.

## Abstract

| Reported value | results.json key |
|---|---|
| n = 1,470; base rate 16.1% | `n`, `base_rate` |
| CV ROC-AUC 0.815 (σ 0.034, CI [0.786, 0.845]) | `cv.auc_mean`, `cv.auc_sd`, `cv.auc_ci` |
| ECE 0.032 (CI [0.023, 0.051]) | `ece_10bin`, `ece_ci` |
| Selected LR 0.827 ± 0.031, Brier 0.095 | `bench.lr_unweighted.auc/.auc_sd/.brier`; rule in `model_selection` |
| Repeated-split range [0.746, 0.880] | `repeated_splits.cb_auc_p2_5_97_5` |
| Ablation 0.815 → 0.793 (−0.022); comp-only 0.680 | `bench.cb_all.auc`, `bench.cb_comp_removed.auc`, `bench.cb_comp_only.auc` |
| Gender EOD 0.109 [0.012, 0.237] | `fair_full.Gender.eod`, `.eod_ci` |
| Age-band EOD 0.353 [0.228, 0.549] | `fair_full.Age band.eod`, `.eod_ci` |
| MI 0.426 / 0.582 nats; z ≈ 49 / 49 | `mi.MaritalStatus.val`, `mi.Age.val`, `mi_null.MaritalStatus.z`, `mi_null.Age.z` |

## Table 1 (study-level comparison)

Qualitative characterization of cited studies [24–32]; no numeric values.

## Table 2 (segmentation) and Figure 2

| Reported value | results.json key |
|---|---|
| KMeans 0.116; PCA(10)+KMeans 0.164; GMM 0.087 | `seg.KMeans`, `seg.PCA10+KMeans`, `seg.GMM` |
| DBSCAN: no ≥2-cluster solution | `seg.DBSCAN_ge2` |
| PCA 10-component variance 69.0% | `pca_var10` |

## Table 3 (predictive benchmark)

Each row maps to one entry of `bench.*` with fields
`auc`, `auc_sd`, `ap`, `brier`, `ece`, `acc`:

| Row | key |
|---|---|
| CatBoost 5:1 (audit subject) | `bench.cb_all` |
| CatBoost none | `bench.cb_all_unweighted` |
| Logistic reg. none (selected) | `bench.lr_unweighted` |
| Logistic reg. balanced | `bench.lr` |
| Random forest balanced | `bench.rf` |
| Random forest none | `bench.rf_unweighted` |
| Majority baseline | `bench.dummy` |
| Compensation removed / only | `bench.cb_comp_removed`, `bench.cb_comp_only` |

Model-selection rule, gate outcomes, and selection: `model_selection`.

## Table 4 (CV summary)

`cv.*` (auc/brier/ap/acc means, SDs, CIs), `ece_10bin`, `ece_ci`,
`fair.Gender.dpd`, `fair.Gender.eod`; DPD/EOD CIs from
`fair_full.Gender.dpd_ci`, `.eod_ci`.

## Overfitting Assessment

| Reported value | results.json key |
|---|---|
| Train AUC 1.00; gap 0.185 | `train_auc`, `overfit_gaps.catboost_weighted.optimism_gap` |
| LR train 0.870, gap 0.043 | `overfit_gaps.logreg_unweighted` |
| RF gaps | `overfit_gaps.rf_balanced`, `overfit_gaps.rf_unweighted` |
| Sweep: CV AUC band 0.811–0.821; train down to 0.925 | `cb_regularization_sweep[*]` |

## Uncertainty subsection

| Reported value | results.json key |
|---|---|
| Conditional bootstrap CI [0.693, 0.859] | `bootstrap_auc_ci` |
| Repeated splits: 0.813 ± 0.035, [0.746, 0.880] | `repeated_splits.cb_auc_mean/.cb_auc_sd/.cb_auc_p2_5_97_5` |
| LR repeated splits 0.833 [0.781, 0.886] | `repeated_splits.lr_auc_mean/.lr_auc_p2_5_97_5` |
| Repeated-split Brier [0.081, 0.116]; ECE [0.033, 0.080] | `repeated_splits.cb_brier_p2_5_97_5/.cb_ece_p2_5_97_5` |
| Holdout AUC 0.781 / Brier 0.108 | `holdout_auc`, `holdout_brier` |

## Calibration subsection; Figures 3–4

`cv.brier_mean`, `ece_10bin`, `ece_adaptive`, `ece_ci`,
`mean_pred_weighted`, `mean_pred_unweighted`,
`ece_10bin_unweighted`, `ece_unweighted_ci`, `holdout_ece`, `cm` (confusion matrix).

## Explainability section; Figures 5–6

| Reported value | results.json key |
|---|---|
| Permutation / SHAP top features | `perm_top`, `shap_top` |
| Spearman ρ 0.24; 0.25 ± 0.05 (0.14–0.33, 10 seeds) | `shap_perm_spearman`, `xai_spearman` |
| Masking top-5 0.070 ± 0.031 vs random 0.018 ± 0.020, z 2.6 | `xai_masking` |
| Shuffled-label AUC 0.486 ± 0.046; max drop 0.025 ± 0.009 | `xai_shuffled_label` |

## Table 5 (fairness) and Figure 7

Per attribute: point estimates `fair.<attr>.*`; subgroup sizes and 95% bootstrap
CIs `fair_full.<attr>.groups/.dir_ci/.cv_ci/.dpd_ci/.eod_ci`.
Selected-model comparison: `fair_lr_unweighted.<attr>`.

## Escalation-Threshold Sensitivity

All sweeps: `threshold_sensitivity.{dir,eod,cramers_v,ece,brier,silhouette}.grid`;
Brier skill 0.245 and base-rate reference 0.135:
`threshold_sensitivity.brier.brier_skill_score/.base_rate_reference`.

## Proxy MI audit

`mi.<attr>` (feature-matrix estimates, top3) and `mi_null.<attr>`
(null-protocol re-estimate `observed`, `null_mean`, `null_sd`, `z`, `p_perm`).

## Figure 8

Panel (a): `seed_aucs`, `seed_min`, `seed_max`. Panel (b): the 100 repeated-split
AUCs summarized in `repeated_splits` (drawn from the same run).

## Figure 9 (dashboard)

Panels reuse: `holdout_auc`, `cv.brier_mean`, `bench.cb_*.auc`, `shap_top`,
`fair.*`, `seed_aucs`.

## Conclusion

All values re-state numbers mapped above.

## Provenance

`provenance`: seed, dataset SHA-256, Python/library versions, generation timestamp.
