# Point-by-Point Response to the Editor and Reviewers (Second Revision)

Manuscript ID: 904dad55-d521-4e72-a8c2-c99d1bc97497
Title: A Governance Aware Framework for Auditing Calibration Explainability and Fairness in HR Attrition Prediction Models
Author: Shaurya Gupta
Journal: Discover Artificial Intelligence (Springer Nature)

Both the Editor and Reviewer 3 located the same underlying problem: the repository accompanying the previous revision did not contain the analysis the manuscript described. The single-pipeline revision had in fact been run before that submission, but the version pushed to GitHub was an older checkout: a `results.json` that predated the revised benchmark, a second and inconsistent results file left over from an earlier pipeline, and a response letter written against scripts that no longer existed. Every numerical discrepancy Reviewer 3 lists below follows from that one release error, not from a problem in the underlying analysis, and we take responsibility for it.

The repository (https://github.com/SilverCoin256/employee-incentive-ai-research) now contains one analysis path. `pipeline.py` (seed 42) reads `data/ibm_hr_attrition.csv` and writes a single output file, `results/results.json`, plus every figure. `TRACEABILITY.md` maps each reported number to its key in that file, and a scripted check, `tools/crosscheck_manuscript.py`, verifies all 208 numeric claims in the manuscript against the JSON at their stated rounding; it passes with zero failures as of this submission. We re-ran the pipeline from a clean checkout immediately before resubmitting: the regenerated file is identical to the committed one except for the run timestamp. The two superseded files (`empirical_run_summary.json`, `arrays.npz`) and the earlier response letter, which cited a different pipeline script and a different primary model, have been removed from the repository.

## Response to the Editor

### Editor, comment 1

> Reconcile every numerical result across the manuscript, response letter and analysis outputs. Complete executable workflow, and ensure that all tables, figures and supplementary results can be traced to this single output.

Resolved as described above. Reviewer 3's specific case, the logistic regression row in Table 3, occurred because the shipped JSON contained only the class-balanced configuration (AUC 0.821, average precision 0.576, Brier 0.162, accuracy 0.755), while the manuscript's selected configuration is the unweighted model. Both weighting configurations for logistic regression, random forest, and CatBoost are now stored (`bench.lr_unweighted`, `bench.lr`, `bench.rf_unweighted`, `bench.rf`, `bench.cb_all_unweighted`, `bench.cb_all`), each with its fold standard deviation and pooled ECE. This letter cites only values that appear under those keys.

### Editor, comment 2

> Define a prespecified and transparent model-selection criterion. CatBoost should not be presented as the primary model merely because it is more complex, particularly where logistic regression performs comparably or better on some measures. The training AUC of 1.00 versus cross-validated AUC of 0.815 also requires a more careful overfitting assessment.

Section 6.1 now states the selection rule before applying it: a calibration gate (pooled cross-validated ECE below 0.10, and CV Brier below the base-rate reference of 0.135, i.e. positive Brier skill), then the highest mean CV AUC among gated configurations, then a parsimony tiebreak when configurations fall within one fold-SD of the best result (logistic regression before random forest before CatBoost). Applied to the six benchmarked configurations, the class-weighted logistic regression fails the gate (ECE 0.200); every remaining configuration sits within one fold-SD of the best AUC (0.827), so the rule selects the unweighted logistic regression for deployment. CatBoost stays in the manuscript only as the model under audit, chosen because gradient-boosted ensembles with native categorical handling are what most deployed HR tools actually use. CatBoost's added complexity buys no discrimination or calibration advantage over the linear model on this benchmark, and the manuscript now says so outright.

On the optimism gap: a new subsection reports it for every model family. CatBoost goes from a training AUC of 1.00 to a cross-validated 0.815, a gap of 0.185; random forest shows a comparable pattern; logistic regression goes from 0.870 to 0.827, a gap of 0.043. A CatBoost regularization sweep across tree depth, iteration count, and l2-leaf-regularization strength drives training AUC down to 0.925 while cross-validated AUC stays inside 0.811–0.821, within one fold-SD of the default configuration. A large drop in training fit paired with a negligible change in held-out performance is the signature of benign interpolation, not variance that would hurt generalization. The manuscript states plainly that training-set metrics carry no evidential weight in the audit; only cross-validated, holdout, and resampled numbers are used to support any claim.

### Editor, comment 3

> Correct the interpretation of the bootstrap interval. Resampling fixed holdout predictions quantifies uncertainty conditional on that fitted model and split; it does not estimate variability across alternative data splits, preprocessing steps and model refitting. Either revise the interpretation or use a nested resampling procedure that repeats the full modelling pipeline.

Both fixes are in. The bootstrap interval [0.693, 0.859] is now labeled everywhere it appears as conditional on the fitted model and the specific 80/20 split, not as an estimate of split-to-split variability. For the latter, we added 100 repeated full-pipeline splits with complete refitting: stratified re-splitting, preprocessing refit, and model retraining on each partition. CatBoost's holdout AUC over those 100 splits averages 0.813 (SD 0.035), with a 2.5–97.5 percentile range of [0.746, 0.880]; logistic regression averages 0.833 [0.781, 0.886]. Figure 8(b) shows the distribution, and the Discussion now attributes split-to-split variability to this procedure rather than to the conditional bootstrap.

### Editor, comment 4

> Make the explainability and fairness analyses fully reproducible. Report the SHAP-permutation agreement, masking controls, shuffled-label analysis and mutual-information null results consistently, with repeated controls and uncertainty estimates. Fairness tables should include subgroup sizes and uncertainty intervals, and all governance thresholds should be justified or clearly described as illustrative escalation rules rather than validated cut-offs.

Each analysis is now computed inside `pipeline.py` with repeated controls and stored under a named key. SHAP–permutation agreement: reference-run Spearman ρ = 0.24, and across ten independent permutation seeds, 0.25 ± 0.05 (range 0.14–0.33), under `xai_spearman`. Masking: jointly permuting the five highest-SHAP features costs 0.070 ± 0.031 AUC over 20 repeats, against 0.018 ± 0.020 for twenty random five-feature sets (z = 2.6), under `xai_masking`. Shuffled-label control: ten refits on permuted labels give AUC 0.486 ± 0.046 and a maximum permutation-importance drop of 0.025 ± 0.009, which the manuscript treats as the evaluation-noise floor, under `xai_shuffled_label`. Mutual-information nulls: 200 permutations per protected attribute, with null mean, SD, z-score, and empirical p-value stored for each, under `mi_null` (MaritalStatus–StockOptionLevel z ≈ 49, p = 0.005; Age–TotalWorkingYears z ≈ 49, p = 0.005; Gender's top feature z = 1.4, p = 0.10, indistinguishable from noise).

Table 5 now reports, per protected attribute, the number of subgroups, their sizes, attrited counts, and 1,000-resample bootstrap 95% CIs for disparate-impact ratio, Cramér's V, demographic parity difference, and equalized odds difference, under `fair_full`. Every governance cut-off is swept rather than asserted (see the response to Reviewer 3, comment 6) and described in the text as an illustrative escalation default rather than a validated threshold.

### Editor, comment 5

> Further moderate the novelty and operational claims. The contribution is best framed as a structured synthesis and proof-of-concept auditing workflow. Claims concerning deployment constraints, vendor benchmarking, quantified business cases and organisational risk should not extend beyond what can be supported using one synthetic benchmark dataset.

The Abstract, Related Work, Discussion, and Conclusion now describe the contribution as a structured methodological synthesis and proof-of-concept auditing workflow, and state directly that no individual pipeline component is new. Table 1 compares against nine identifiable studies [24–32] rather than author-defined categories (see the response to Reviewer 3, comment 1). Operational language has been cut back throughout: the vendor-comparison discussion is now called an illustrative vendor-audit workflow, "a workflow sketch… not a validated vendor-comparison instrument"; the ablation results are described as demonstrations of the auditing workflow on synthetic data rather than quantified business cases; and the deployment-template paragraph now states that its outputs inform, not certify, documentation, with regulatory compliance requiring validation on real organizational data. A Declarations section (Ethical approval; Consent to participate; Consent to publish) has been added.

## Response to Reviewer 1

We are grateful for the recommendation to accept.

## Response to Reviewer 3

We thank the reviewer for the specificity of these comments, which made the underlying release problem easy to locate and fix.

### Reviewer 3, comment 1

> The novelty claim is still not adequately supported. The literature comparison relies on broad, author-defined categories rather than identifiable studies. Unless the claimed gap is demonstrated through a transparent study-level comparison, the contribution should be framed as a methodological synthesis rather than as the resolution of an established research gap.

Table 1 was rebuilt around identifiable studies: seven empirical attrition-prediction papers that use the same IBM benchmark (Zhao and Hryniewicki; Jain and Nayyar; Fallucchi et al.; Qutub et al.; Al-Darraji et al.; Raza et al.; Guerranti and Dimitri [24–30]) and two fairness-oriented analyses of algorithmic HR tools (Raghavan et al.; Köchling and Wehner [31, 32]), scored per evaluation component with the characterization method stated in the caption. We now name the gap precisely in the surrounding text: not the absence of any one technique, but the absence of their co-designed integration across the surveyed studies. The contribution is described as a methodological synthesis throughout, not as the resolution of an established research gap.

### Reviewer 3, comment 2

> Important numerical inconsistencies remain between the manuscript and the supplied results file. For example, the Logistic Regression results reported in Table 3 differ substantially from those in results.json, particularly for average precision, Brier score, and accuracy. These differences cannot be explained by rounding. All tables, figures, and reported values should be verified against one fixed analysis output.

This traces to the release error described at the top of this letter. Both logistic-regression configurations are now in `results.json` with the same structure as every other benchmarked model: fold SD, CI, pooled ECE. A scripted check, `tools/crosscheck_manuscript.py`, verifies all 208 numeric claims in the manuscript against that file; we ran it after finishing this revision and it passes with no failures.

### Reviewer 3, comment 3

> The selection of CatBoost as the primary model remains unclear. Logistic Regression performs better on several metrics reported in the manuscript, while equivalent confidence intervals and calibration results are not provided for all models. A prespecified model-selection criterion is needed. In addition, the training AUC of 1.00 versus a cross-validated AUC of 0.815 represents a substantial optimism gap that requires a more rigorous overfitting assessment.

Addressed in the response to Editor comment 2 above: a prespecified, outcome-independent rule now selects the unweighted logistic regression for deployment, CatBoost remains only as the audit-demonstration subject, identical fold-based intervals and pooled ECE are reported for every configuration in Table 3, and a new subsection reports per-family optimism gaps together with the CatBoost regularization sweep.

### Reviewer 3, comment 4

> The newly reported XAI validation results are not fully traceable. The response states that all reported values are stored in results.json; however, the top-feature versus random-feature masking effects and the shuffled-label AUC are not included in the supplied file. The SHAP–permutation Spearman coefficient also differs between the response letter, manuscript, and results.json. These analyses should be documented and reported consistently, including repeated random controls and uncertainty estimates.

All four analyses are now in `results.json` (see the response to Editor comment 4). On the Spearman coefficient specifically: the manuscript reports both the single reference-run value (0.24) and the ten-seed distribution (0.25 ± 0.05, range 0.14–0.33). The 0.27 that appeared in the earlier response letter came from the superseded pipeline and does not appear anywhere in the current submission.

We also traced a related discrepancy the reviewer may have noticed: an earlier internal draft reported a gender-proxy z-score of roughly 3.8 and an age z-score of roughly 39, against 1.4 and 49 in the current manuscript. These came from two different mutual-information protocols, a feature-matrix estimate versus a single-feature null re-estimate, with age treated once as banded and once as continuous. We fixed one protocol throughout (single-feature KSG re-estimates under a 200-permutation null, age treated as continuous) and now report both the feature-matrix value and the null-protocol re-estimate for each attribute, under `mi` and `mi_null`.

### Reviewer 3, comment 5

> The bootstrap confidence interval is based on resampling fixed holdout predictions without model refitting. It therefore does not represent performance under alternative train–test splits or repeated model training. The interpretation should be corrected, or a resampling procedure that includes data splitting, preprocessing, and model refitting should be used.

Addressed in the response to Editor comment 3 above.

### Reviewer 3, comment 6

> The threshold analysis remains incomplete. Only the Cramér's V trigger is examined for sensitivity, whereas the ECE, Brier, silhouette, equalized-odds, and disparate-impact cut-offs remain insufficiently justified. Fairness results should also include subgroup sizes and uncertainty intervals rather than pooled point estimates alone.

Every cut-off is now swept, not only Cramér's V (`threshold_sensitivity`). ECE: the audit subject passes at every threshold from 0.04 upward. Brier: the gate anchors on positive Brier skill against the base-rate reference (0.245) rather than a fixed cut-off, because a fixed bound of 0.15 would be weaker than the base rate itself (0.135). Silhouette: the segmentation-not-warranted finding holds for any threshold at or above 0.17 (the maximum observed silhouette is 0.164). Equalized odds: the flag set is stable from 0.20 to 0.35 (age band, compensation band); marital status joins at 0.15; gender joins at 0.10. Disparate impact: the flag set is unchanged across 0.60–0.90. Cramér's V: the compensation-band flag persists from 0.10 to 0.25 and clears only at 0.30; gender is never flagged at any value tested.

Table 5 now carries subgroup counts, attrited counts, and bootstrap CIs for every metric and attribute, and we added the same fairness battery for the selected logistic regression (`fair_lr_unweighted`). The escalation set changes under that model: compensation-band Cramér's V falls to 0.177, below the default threshold, while marital-status equalized odds rises to 0.221, above it, reversing the pattern seen under CatBoost. The manuscript now reports this as a finding rather than an inconsistency to explain away: fairness escalation is a property of the fitted model, not of the dataset alone, and an audit needs to be re-run against whichever model is actually deployed.

### Reviewer 3, comment 7

> Although the proof-of-concept framing has improved, several proxy-risk and operational claims remain too strong for a single synthetic dataset. The response reports an unweighted ECE, feature-masking effects, shuffled-label results, and mutual-information permutation-null statistics, yet these values are not contained in the supplied results.json despite the statement that all reported values are stored there. These results should be made fully traceable, and terms such as "deployment constraint," "vendor-evaluation benchmark," "quantified business case," and "organizational risk estimates" should be further moderated.

The unweighted ECE (0.052, CI [0.041, 0.071]), the masking effects, the shuffled-label results, and the MI permutation-null statistics are now all stored in `results/results.json` under the keys listed above. They existed in an internal run at the time of the last response letter but were not included in what was pushed to GitHub. That gap is the failure this letter opens with. On the operational language: "deployment constraint" no longer appears in the manuscript; the vendor comparison is called a workflow sketch rather than a benchmark; "quantified business case" appears only in the negation ("not quantified business cases"); and organizational-risk claims are bounded, in the Scope Conditions section, to pipeline mechanics demonstrated on one synthetic dataset.
