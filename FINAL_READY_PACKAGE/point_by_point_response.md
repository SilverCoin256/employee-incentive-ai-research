# Point-by-Point Response to the Editor and Reviewers (Second Revision)

**Manuscript ID:** 904dad55-d521-4e72-a8c2-c99d1bc97497
**Title:** *A Governance Aware Framework for Auditing Calibration Explainability and Fairness in HR Attrition Prediction Models*
**Author:** Shaurya Gupta
**Journal:** Discover Artificial Intelligence (Springer Nature)

We thank the Editor and Reviewer 3 for identifying that the previous revision's central weakness was not presentational but infrastructural: the public repository did not contain the analysis outputs that the manuscript claimed to trace to. We state plainly what went wrong and what has changed.

**What went wrong.** The revised single-pipeline analysis had been prepared, but the repository release accompanying the previous revision shipped superseded round-1 artifacts: a `results.json` that predated the revised benchmark (containing only the class-weighted logistic regression), a second results file (`empirical_run_summary.json`) from an earlier pipeline, and a response letter referencing scripts that no longer existed. Every numerical discrepancy Reviewer 3 identified follows from this release-management failure, and we take full responsibility for it.

**What has changed.** The repository now contains exactly one analysis path and one output:

- `pipeline.py` (seed 42) reads `data/ibm_hr_attrition.csv` and writes `results/results.json` plus every figure. One command regenerates everything.
- `results/results.json` is the **only** results file; `empirical_run_summary.json` and `arrays.npz` have been removed.
- `TRACEABILITY.md` maps every reported value — abstract, all tables, all figures, all in-text statistics — to a named key in `results/results.json`.
- We re-ran the pipeline from a clean checkout before this resubmission: the regenerated `results.json` is byte-identical to the committed file except the timestamp. We additionally ran a scripted 208-point cross-check of every numeric claim in the manuscript against the JSON keys; all pass at the stated rounding.
- A provenance block (`provenance`) records the seed, the dataset SHA-256, and all library versions.

---

## Editor's Comments

### E1. "Reconcile every numerical result across the manuscript, response letter and analysis outputs. Complete executable workflow…"

Done, as summarized above. Specifics of the previously reported discrepancies:

- **Logistic regression (Table 3 vs old results.json).** The manuscript's selected configuration is the *unweighted* logistic regression; the previously shipped JSON contained only the class-*balanced* configuration (AUC 0.821, AP 0.576, Brier 0.162, Acc 0.755). Both configurations are now benchmarked and stored with fold SDs and pooled ECE: `bench.lr_unweighted` (AUC 0.827 ± 0.031, AP 0.614, Brier 0.095, ECE 0.027, Acc 0.876) and `bench.lr` (the balanced row). The same dual-configuration treatment applies to random forest (`bench.rf`, `bench.rf_unweighted`) and CatBoost (`bench.cb_all`, `bench.cb_all_unweighted`).
- **Response letter.** This response contains no number that is not a key in `results/results.json`; the stale round-1 response has been removed from the repository.

### E2. "Define a prespecified and transparent model-selection criterion… The training AUC of 1.00 versus cross-validated AUC of 0.815 also requires a more careful overfitting assessment."

- **Prespecified rule (Section 6.1).** Stated in outcome-independent terms and applied uniformly: (i) calibration gate — pooled CV ECE < 0.10 and CV Brier below the base-rate reference (0.135, positive Brier skill); (ii) among gated configurations, highest mean CV ROC-AUC; (iii) parsimony — configurations within one fold-SD of the best resolve to the least complex family (LR < RF < CatBoost). Applied to Table 3, the balanced LR fails the gate (ECE 0.200); all remaining configurations lie within one fold-SD of the best AUC (0.827), so the rule selects the **unweighted logistic regression** as deployment-preferred. The rule and its intermediate outputs are logged verbatim in `model_selection`.
- **CatBoost's role.** CatBoost is retained only as the audit-demonstration subject (the model class prevalent in commercial HR tooling); the manuscript now states explicitly that its complexity buys no discrimination or calibration advantage on this benchmark — itself a governance finding.
- **Overfitting assessment (new subsection).** Optimism gaps for every family (`overfit_gaps`): CatBoost train 1.00 vs CV 0.815 (gap 0.185); RF train ≈1.00, CV 0.807–0.810; LR train 0.870, CV 0.827 (gap 0.043). A CatBoost regularization sweep (`cb_regularization_sweep`: depth 3–6, 100–400 iterations, l2_leaf_reg 3–10) moves training AUC from 1.00 to 0.925 while CV AUC stays within 0.811–0.821 — under one fold-SD of the default — indicating benign interpolation rather than variance harming generalization. The manuscript also states the governance rule that training-set metrics carry no evidential weight in the audit.

### E3. "Correct the interpretation of the bootstrap interval… or use a nested resampling procedure that repeats the full modelling pipeline."

Both corrections were made:

- The bootstrap CI [0.693, 0.859] is now described only as uncertainty **conditional on the fitted model and the specific 80/20 split** (Sections 3, 6.4, Discussion, Scope Conditions).
- We added **100 repeated full-pipeline splits with complete refitting** (stratified re-splitting, preprocessing re-fit, model re-training per split; `repeated_splits`): CatBoost holdout AUC 0.813 ± 0.035, 2.5–97.5 percentile range [0.746, 0.880]; logistic regression 0.833 [0.781, 0.886]; repeated-split Brier [0.081, 0.116] and ECE [0.033, 0.080]. Figure 8(b) shows the distribution. The manuscript's governance conclusion now attributes split-to-split variability to this procedure, not to the conditional bootstrap.

### E4. "Make the explainability and fairness analyses fully reproducible… repeated controls and uncertainty estimates… subgroup sizes and uncertainty intervals… thresholds justified or clearly described as illustrative."

All of the following are now computed inside `pipeline.py`, reported in the manuscript with uncertainty, and stored under named keys:

- **SHAP–permutation agreement** (`xai_spearman`): reference run ρ = 0.24; across ten permutation seeds, 0.25 ± 0.05 (range 0.14–0.33).
- **Masking controls** (`xai_masking`): jointly permuting the five highest-SHAP features degrades AUC by 0.070 ± 0.031 (20 repeats) versus 0.018 ± 0.020 for twenty random five-feature sets (z = 2.6).
- **Shuffled-label control** (`xai_shuffled_label`): ten refits on permuted labels, AUC 0.486 ± 0.046, maximum permutation-importance drop 0.025 ± 0.009, reported as the evaluation-noise floor.
- **Mutual-information permutation nulls** (`mi_null`): 200 permutations per attribute; null mean, SD, z, and empirical p stored (MaritalStatus–StockOptionLevel z ≈ 49, p = 0.005; Age–TotalWorkingYears z ≈ 49, p = 0.005; Gender's top feature z = 1.4, p = 0.10, i.e., indistinguishable from the null).
- **Fairness tables** (`fair_full`): Table 5 now reports, per attribute, the number of subgroups, subgroup sizes, attrited counts, and 1,000-resample bootstrap 95% CIs for DIR, Cramér's V, DPD, and EOD.
- **Thresholds**: every governance cut-off is described as an illustrative escalation default, justified where a source exists (four-fifths rule for DIR; positive Brier skill versus the base-rate forecast for the calibration gate; decision-support practice for ECE < 0.10), and swept in `threshold_sensitivity` (see E-comment 4 / Reviewer 3 point 6 below).

### E5. "Further moderate the novelty and operational claims."

- The contribution is framed throughout (Abstract, Related Work, Discussion, Conclusion) as a **structured methodological synthesis and proof-of-concept auditing workflow**; the manuscript states that no individual component is new.
- Table 1 was rebuilt as a study-level comparison against identifiable publications [24–32] rather than author-defined categories, with the characterization method stated in the caption.
- Operational language was cut back to workflow illustrations: "illustrative vendor-audit workflow… a workflow sketch, not a validated vendor-comparison instrument"; ablation results are "demonstrations of the auditing workflow on synthetic data, not quantified business cases"; pre-deployment outputs "inform — not certify" documentation and "regulatory compliance requires validation on real organizational data."
- The requested **Declarations** section (Ethical approval; Consent to participate; Consent to publish) has been added.

---

## Reviewer 1

We thank the reviewer for recommending acceptance.

## Reviewer 3

### R3-1. Novelty demonstrated through study-level comparison

Table 1 now lists identifiable studies — seven empirical attrition-prediction papers on the same IBM benchmark (Zhao et al.; Jain & Nayyar; Fallucchi et al.; Qutub et al.; Al-Darraji et al.; Raza et al.; Guerranti & Dimitri [24–30]) and two fairness-oriented analyses of algorithmic HR tools (Raghavan et al.; Köchling & Wehner [31,32]) — scored per evaluation component (discrimination, calibration, uncertainty, attribution validation, fairness, thresholds) with ●/○/— codes and the basis of characterization stated in the caption. The surrounding text now frames the contribution as a methodological synthesis whose claimed gap is "not the absence of any single technique but the absence of their co-designed integration" in the surveyed studies.

### R3-2. Numerical inconsistencies between manuscript and results file

Resolved at the root; see E1. The specific example the reviewer cites (Table 3 LR vs `results.json`) arose because the shipped JSON predated the revised benchmark and contained only the class-weighted LR; both weighting configurations are now stored with identical structure, and the 208-point automated cross-check verifies every table, figure, and in-text value against `results/results.json`.

### R3-3. CatBoost selection and the optimism gap

See E2: a prespecified, outcome-independent selection rule now governs model choice and selects the unweighted logistic regression; CatBoost is retained solely as the audit-demonstration subject; equivalent fold-based intervals and pooled ECE are reported for **all** benchmarked configurations; and a dedicated overfitting subsection reports per-family optimism gaps plus the regularization sweep showing CV AUC is insensitive (0.811–0.821) while training AUC drops to 0.925.

### R3-4. Traceability of XAI validation results

All four analyses the reviewer names are now computed in `pipeline.py` and stored: masking effects (`xai_masking`), shuffled-label control (`xai_shuffled_label`), SHAP–permutation Spearman (`xai_spearman`), and MI nulls (`mi_null`) — each with repeated controls and dispersion, as detailed under E4. On the previously inconsistent Spearman coefficient: the manuscript now reports the reference-run value (0.24) together with the ten-seed distribution (0.25 ± 0.05, range 0.14–0.33); the earlier response letter's 0.27 came from the superseded pipeline and no longer appears anywhere. Where earlier drafts reported a gender-proxy z ≈ 3.8 and an age z ≈ 39, the discrepancy traced to protocol differences (feature-matrix estimate vs single-feature null re-estimate; banded vs continuous age); the manuscript now fixes one protocol — single-feature KSG re-estimates under a 200-permutation null, age treated as continuous — reports both the feature-matrix value and the null-protocol re-estimate, and stores all of it under `mi` and `mi_null`.

### R3-5. Bootstrap interpretation

See E3. The conditional interpretation is now stated wherever the bootstrap CI appears, and the added 100-split full-refit procedure (`repeated_splits`) provides the across-split estimate the reviewer requested, reported side-by-side and attributed correctly.

### R3-6. Threshold analysis and fairness reporting

Every cut-off is now swept, not only Cramér's V (`threshold_sensitivity`):

- **ECE**: audit subject passes at every threshold from 0.04 upward (fails only at ≤ 0.03).
- **Brier**: the gate anchors on positive Brier skill vs the base-rate reference (BSS = 0.245) because a fixed 0.15 cut-off would be weaker than the base-rate forecast itself (0.135).
- **Silhouette**: the segmentation-not-warranted finding is stable for any threshold ≥ 0.17 (maximum observed 0.164).
- **EOD**: flag set stable from 0.20 to 0.35 (age band, compensation band); marital status joins at 0.15; gender joins at 0.10.
- **Disparate impact**: sweeping 0.60–0.90 never changes the flag set.
- **Cramér's V**: compensation-band flag persists from 0.10 to 0.25, clears only at 0.30; gender is never flagged.

Table 5 now includes subgroup counts, attrited counts per subgroup, and bootstrap CIs for every metric (see E4). We additionally report the same fairness battery for the selected logistic regression (`fair_lr_unweighted`): the escalation set changes (compensation-band V falls to 0.177, below threshold; marital-status EOD rises to 0.221, above threshold), which the manuscript surfaces as a governance finding — fairness flags are properties of the fitted model and must be re-run against the deployed model.

### R3-7. Proxy-risk and operational claims; traceability of reported controls

The unweighted ECE (0.052, CI [0.041, 0.071]), masking effects, shuffled-label results, and MI permutation-null statistics are all now stored in `results/results.json` under the keys listed in E4 — the earlier statement that they were stored was true of an internal run but not of the shipped file, which is the failure described at the top of this letter. The operational vocabulary has been rewritten: "deployment constraint" framing is gone; the vendor comparison is an "illustrative vendor-audit workflow… not a validated vendor-comparison instrument"; "quantified business cases" appears only in the negation ("not quantified business cases"); and organizational-risk claims are bounded by the Scope Conditions section to pipeline mechanics on one synthetic benchmark.

---

## Summary of changes to the repository

| Item | Previous review state | Current state |
|---|---|---|
| Results files | `results.json` (stale) + `empirical_run_summary.json` + `arrays.npz` | `results/results.json` only |
| Pipeline | `pipeline.py` reading a nonexistent path; older scripts referenced in letters | one `pipeline.py`, runs from repo root, regenerates everything |
| LR/RF/CatBoost configs | one weighting each | both weightings, each with fold SD, CI, pooled ECE |
| Selection rule | absent | `model_selection` (gate → max AUC → parsimony), logged |
| Overfitting | train/CV gap only | `overfit_gaps` + `cb_regularization_sweep` |
| Split variability | conditional bootstrap only | + `repeated_splits` (100 full refits) |
| XAI controls | not in shipped file | `xai_spearman`, `xai_masking`, `xai_shuffled_label` |
| MI nulls | not in shipped file | `mi_null` (200 permutations, z, empirical p) |
| Fairness | pooled point estimates | subgroup n, attrited counts, bootstrap CIs; LR parallel |
| Thresholds | Cramér's V sweep only | all six cut-offs swept (`threshold_sensitivity`) |
| Traceability | claimed | `TRACEABILITY.md` + clean-checkout re-run verified byte-identical |
