# Point-by-Point Response to Reviewers

**Manuscript ID:** 904dad55-d521-4e72-a8c2-c99d1bc97497
**Title (revised):** *A Governance Aware Framework for Auditing Calibration Explainability and Fairness in HR Attrition Prediction Models*
**Author:** Shaurya Gupta
**Journal:** Discover Artificial Intelligence (Springer Nature)

We thank the Editor and the three reviewers for their careful and constructive reading. The comments converged on one central issue — the internal consistency and reproducibility of the code, figures, and reported numbers — and we have addressed it structurally rather than cosmetically.

## Summary of the principal change

All analyses now run through a **single end-to-end pipeline**, `run_real_empirical_pipeline.py`, which emits every scalar metric to `results/empirical_run_summary.json` and every raw prediction array to `results/arrays.npz`. `generate_figures.py` now reads **only** those artifacts, so every figure is drawn directly from raw model predictions and computed metrics. In the prior submission the figure script used summary/illustrative values embedded in the plotting code rather than the pipeline outputs; we recognise this broke traceability, and it has been removed. As a result:

- The holdout, cross-validation, bootstrap, seed-sensitivity, SHAP, permutation, fairness, proxy-MI, and clustering results now originate from one code path and one model version (Reviewer 3, points 2–3).
- Several numbers changed modestly when regenerated from the unified pipeline with the corrected preprocessing and feature set. Where they changed, the manuscript, tables, and figures have all been updated to the pipeline outputs. The qualitative conclusions are unchanged.

The primary model is CatBoost (400 iterations, learning rate 0.05, depth 6, native categorical handling, positive-class weight 5:1), with protected attributes (gender, age, marital status) excluded from the predictor set and retained only for fairness auditing. The primary performance estimate is the stratified 5-fold cross-validation mean; the single holdout split, bootstrap interval, and seed sweep are reported as robustness checks.

---

## Editor comments

**E1. Title without punctuation, readable as one concise sentence.**
Revised to: *"A Governance Aware Framework for Auditing Calibration Explainability and Fairness in HR Attrition Prediction Models"* — no colon, hyphen, or other punctuation, retaining the key terms of the original title.

**E2. Declarations section with individual subheadings.**
A **Declarations** section now contains the three requested subheadings, each addressed separately:
- *Ethical approval* — Not applicable. The study uses a publicly available synthetic dataset (IBM HR Analytics) and involves no human participants, human tissue, or animals.
- *Consent to participate* — Not applicable. No human participants are involved.
- *Consent to publish* — Not applicable. No individual person's data are reported.

**E-summary 1 (stage ordering).** See Reviewer 1, point 1.
**E-summary 2 (reconcile AUC/ECE, one primary).** See Reviewer 1, point 2.
**E-summary 3 (calibration uncertainty).** See Reviewer 1, point 3.
**E-summary 4 (traceability/consistency).** See the Summary above and Reviewer 3, points 2–8.
**E-summary 5 (moderate claims; proof-of-concept).** See Reviewer 3, point 13; the abstract, Discussion, and Scope Conditions now frame the study as a proof-of-concept demonstrating pipeline mechanics on a synthetic benchmark, with enterprise/regulatory/proxy claims moderated accordingly.

---

## Reviewer 1

**R1.1 — Why must calibration precede fairness auditing? State which downstream interpretations are invalidated if an upstream stage fails, with a concrete example.**
We agree the ordering was asserted rather than argued, and we have narrowed the claim. We now distinguish two kinds of dependency:
- *Genuine inter-stage dependencies.* The calibration stage governs whether the model's outputs may be read as **probabilities** (risk scores for budgeting/retention triage). Demographic-parity and equalized-odds statistics are computed from **thresholded** predictions and, as the reviewer correctly notes, do **not** require calibrated probabilities. We now state this explicitly. The dependency that does hold is: (i) point-estimate stability must be established before any subgroup gap is interpreted as a property of the model rather than of the split — with 47 positives in a holdout, a subgroup EOD can swing across resamples; we therefore report fairness from pooled cross-validation and show its threshold sensitivity. (ii) Calibration is a prerequisite for the *probability-based* governance use (risk scoring), not for the *classification-parity* audit.
- *Concrete example of a wrong conclusion under reordering.* If one reports probability-based risk tiers before checking calibration, a systematically over-confident model (e.g., the 5:1-weighted configuration inflates mean predicted probability from 0.116 to 0.174 against a 0.161 base rate) would assign inflated absolute attrition risk, over-allocating retention budget — even though its ranking (AUC) and its thresholded subgroup gaps are unchanged. The text now frames the pipeline as a checklist with two genuine dependencies rather than a strict total order.

**R1.2 — Reconcile the multiple AUC values; define one primary result; present others as robustness checks. Same for ECE.**
Done. There is now **one primary AUC**: the stratified 5-fold cross-validation mean, **0.810 ± 0.019 (95% CI [0.793, 0.827])** for the primary CatBoost configuration. The single 80/20 holdout (AUC 0.783), its bootstrap interval ([0.695, 0.854]), and the 7-seed sweep (0.773–0.836, mean 0.805) are now explicitly labelled robustness checks that quantify split sensitivity. For ECE, the **primary figure is 0.043 (10-bin uniform, 95% CI [0.031, 0.061])** for the primary model; the unweighted configuration (0.050 [0.039, 0.069]) and the adaptive-binning estimate (0.048) are reported as sensitivity checks. The abstract, results, and captions now use these primary values consistently.

**R1.3 — Calibration needs uncertainty treatment; justify or replace 10 uniform bins given few positives.**
Implemented. ECE is now reported with a **bootstrap 95% CI** (1000 resamples of the pooled test predictions) and with an **equal-count (adaptive) binning** estimator alongside the uniform-bin estimate, precisely because the ~237 positives spread thinly across ten uniform bins make the uniform estimate unstable. Uniform-bin ECE 0.043 [0.031, 0.061] and adaptive-bin ECE 0.048 [0.034, 0.065] agree, so the calibration claim is now stated with its uncertainty rather than as a point value.

**R1.4 — "Nearly 15%" conflates a 0.149 absolute AUC drop with a percentage.**
Corrected, and the underlying analysis was rebuilt as a clean leave-group-out ablation (see R3.5). We no longer use a percentage. The text now states the absolute comparison directly: compensation-only AUC = 0.671 vs full-feature AUC = 0.810 (a 0.140 absolute-AUC difference), and — importantly — **removing** the compensation group from the full model lowers AUC by only **0.017** (0.810 → 0.793), indicating that compensation is largely redundant with other structural features rather than the dominant signal. This corrects the earlier framing.

**R1.5 — MI proxies reported without units, estimator, or null.**
Fixed. The audit now specifies the estimator (**scikit-learn `mutual_info_classif`, Kraskov k-NN with k = 3, values in nats**) and adds a **permutation null** (200 label shuffles per attribute). The headline proxies are reported against that null: MaritalStatus ↔ StockOptionLevel MI = 0.426 nats (null mean 0.006, 95th pct 0.024; z ≈ 48); AgeBand ↔ TotalWorkingYears MI = 0.386 (z ≈ 39); Gender's top feature MI = 0.031 (z ≈ 3.8, below the 0.15 flag). Readers can now judge significance against the null.

**R1.6 — Bootstrap CI procedure unspecified.**
Specified. The interval is a **percentile bootstrap over 1000 paired resamples of the holdout test predictions** (resampling (y, p̂) pairs; resamples with a single class are discarded). This is now stated in the text and the figure caption; the resulting interval is [0.695, 0.854].

---

## Reviewer 2

**R2.1 — Map each method in the pipeline sentence to its manuscript location.**
Added an explicit mapping. Leakage-controlled preprocessing → §4 (Dataset Governance and Preprocessing); comparative classification with feature ablation → §6.1 and Table 3; ECE analysis → §6.4 and Fig. 4; bootstrap uncertainty → §6.3 and Fig. 8; multi-attribute subgroup fairness → §8, Table 5, and Fig. 7. A one-line pointer to each section now follows the sentence in the Introduction.

**R2.2 — Future Work rationale is too extensive; justify the selection.**
Trimmed from six items to the three that follow directly from the study's limitations: real-workforce validation (the synthetic-data limitation), temporal robustness (calibration drift), and counterfactual explanation (contestability). The remaining items were removed to keep Future Work tied to demonstrated gaps.

**R2.3 — Is the IBM dataset simulated or real?**
It is **simulated (synthetic)**. IBM created it as a fictional dataset for analytics education; it does not describe real employees. This is now stated plainly at first mention in §4 and in the Scope Conditions, and it is the basis for framing the study as a proof-of-concept.

---

## Reviewer 3

**R3.1 — Limited novelty; literature comparison not systematic.**
We have moderated the novelty claim. The contribution is explicitly positioned as an **integration and reproducibility contribution** — a single auditable template that couples calibration-with-uncertainty, leakage-controlled ablation, XAI with faithfulness checks, and multi-attribute fairness with a proxy-MI null — rather than a new method. The positioning table (Table 1) is now described as an illustrative scoping comparison based on cited methodology reviews, not a systematic review, and the text says so.

**R3.2 — Repository does not reproduce all results through one pipeline.**
Resolved. `run_real_empirical_pipeline.py` now produces the holdout, cross-validation, bootstrap, seed-sensitivity, calibration, XAI, fairness, proxy-MI, and clustering results in a single run from one model version, writing `results/empirical_run_summary.json` and `results/arrays.npz`. Figures are regenerated from those files.

**R3.3 — Figures generated from fixed/mock values rather than raw predictions.**
Resolved. Every figure is now generated by `generate_figures.py` strictly from `results/arrays.npz` (raw predictions, SHAP/permutation arrays) and `results/empirical_run_summary.json`. ROC and calibration curves are computed from the actual predictions; SHAP, permutation, fairness, proxy-MI, seed, and clustering panels read the pipeline's stored outputs. No values are embedded in the plotting code.

**R3.4 — Age described as excluded but appears as a SHAP feature.**
Resolved. Age is now unambiguously treated as a **protected attribute excluded from the predictor set** and used only for fairness auditing (as an age band). Consequently Age no longer appears in the SHAP or permutation results, which are recomputed from the actual model feature space. The revised importance figures contain no protected attributes.

**R3.5 — Ablation design unclear (salary-only vs salary-removed vs with/without).**
Resolved with a clearly defined **leave-group-out** design on identical splits and settings. The compensation group is stated explicitly (MonthlyIncome, MonthlyRate, DailyRate, HourlyRate, PercentSalaryHike, StockOptionLevel, and the two compensation-derived indices). We report full-feature AUC 0.810, compensation-**removed** AUC 0.793 (Δ = 0.017), and compensation-**only** AUC 0.671 (Δ from full = 0.140). Table 3 and §6.1 now use this consistent terminology.

**R3.6 — Preprocessing leakage: one-hot on the full dataset before CV.**
Resolved. Encoding, scaling, and imputation are now fit **inside each training fold** via scikit-learn `Pipeline`/`ColumnTransformer` for the linear and forest models; CatBoost uses native categorical handling (no global one-hot). Only the row-wise, parameter-free structural indices are computed before splitting, and the text notes they introduce no fitted parameters and hence no leakage.

**R3.7 — Model definitions and hyperparameters insufficiently reported; "default CatBoost" conflicts with the code.**
Fixed. All settings are now reported: CatBoost (iterations 400, learning rate 0.05, depth 6, positive-class weight 5:1); Logistic Regression (L2, C = 1.0, standardized inputs, max_iter 2000); Random Forest (400 trees, min_samples_leaf 2); Dummy (prior/stratified baseline). The manuscript no longer describes CatBoost as "default."

**R3.8 — CatBoost rationale (native categoricals) inconsistent with one-hot encoding.**
Resolved by using CatBoost's **native categorical handling** (the categorical columns are passed as `cat_features`, not one-hot encoded). The stated rationale now matches the implementation; one-hot encoding is used only for the linear/forest baselines that require it.

**R3.9 — Overfitting not evaluated.**
Added. We now report the **train–test AUC gap**: training AUC 1.00 vs cross-validated test AUC 0.810 (gap 0.19), which we discuss openly as optimism from an unregularized tree depth. The cross-validation and holdout estimates are the unbiased performance figures used throughout; we note depth/early-stopping tuning as a mitigation for deployment. Random Forest shows the same train-AUC saturation, consistent with tree ensembles on this data.

**R3.10 — XAI results not validated.**
Added faithfulness, stability, and sanity checks; the XAI outputs are now presented as **exploratory model-inspection** evidence. Specifically: (i) SHAP-vs-permutation rank agreement (Spearman ρ = 0.27); (ii) a **deletion/faithfulness** curve — masking the top SHAP features degrades AUC substantially more than masking random features (mean gap 0.044); (iii) a **sanity check** — a model trained on permuted labels yields maximum permutation importance ≈ 0.02 (near zero). These support using SHAP/permutation for inspection while cautioning against causal reading.

**R3.11 — Calibration claims too strong; class-weight effect not examined.**
Examined directly. We now report calibration for both the **5:1-weighted** primary model and the **unweighted** model. The weighting shifts the mean predicted probability from 0.116 (unweighted) to 0.174 against a 0.161 base rate, while ECE remains within the governance bound under both settings (0.043 weighted, 0.050 unweighted; both < 0.10). We therefore state that the weighting modestly inflates absolute probabilities but does not break bin-level calibration, and we present risk-score usability with this caveat and the ECE confidence intervals. All calibration language is softened and tied to the synthetic benchmark.

**R3.12 — Governance thresholds unjustified; no sensitivity analysis.**
Added a threshold-sensitivity analysis. For each escalation threshold (EOD, Cramér's V, disparate-impact) we now report which attributes are flagged as the threshold varies (e.g., compensation band is flagged across the full range of Cramér's V thresholds 0.10–0.25; gender is never flagged except at the loosest settings). The thresholds are described as **operational review triggers**, not validated cut-offs, and their sensitivity is shown so readers see the flag stability.

**R3.13 — Fairness/proxy/practical implications over-interpreted.**
Moderated throughout. Raw subgroup metrics and MI values from a synthetic dataset are now described as pipeline-mechanics demonstrations, not evidence of real-world proxy risk. "Severe"/"high-risk" language is replaced with "flagged for review." Claims about enterprise applicability, regulatory compliance, and operational usefulness are reframed as proof-of-concept outputs requiring validation on real HR data with appropriate governance, consistent with the Scope Conditions and Editor comment 5.

---

We believe these revisions resolve the traceability and consistency concerns at their root and place the empirical claims on a fully reproducible footing. All code, artifacts, and figures are available in the public repository. We are grateful for the reviewers' diligence, which materially improved the rigour of the work.
