# Final Validation Report

## Source validation

- **LaTeX source located**: `submission_exports/final_submission_package/final_submission_manuscript.tex`
- **Verified bibliography located**: `submission_exports/final_submission_package/references_verified.bib`
- **Figures folder located**: `submission_exports/final_submission_package/figures/`
- **Tables folder located**: `submission_exports/final_submission_package/tables/`

## Figure validation

The final package figure folder contains all eight figure files referenced by the LaTeX manuscript:

- **`calibration_curve.png`**
- **`cluster_profile_heatmap.png`**
- **`clustering_silhouette_comparison.png`**
- **`confusion_matrix_best_model.png`**
- **`fairness_gender_predicted_positive_rate.png`**
- **`permutation_importance_best_model.png`**
- **`seed_sensitivity_logistic_auc.png`**
- **`shap_summary_best_model.png`**

## Bibliography validation

- **Verified BibTeX entries**: 19 entries in `references_verified.bib`.
- **Duplicate Ribeiro entry removed**.
- **Barocas and Selbst DOI corrected to SSRN version**.
- **IBM HR Kaggle dataset entry included**.

## Scientific-safety validation

The final manuscript retains the required conservative boundaries:

- **No causal overclaims**: model outputs are described as predictive/correlational.
- **No validated-archetype language**: clustering is explicitly exploratory and structurally limited.
- **No deployment-readiness claim**: organizational use is conditioned on external validation, oversight, mitigation, and monitoring.
- **Fairness caveats explicit**: diagnostics are not treated as compliance, mitigation, or fairness guarantees.
- **Synthetic benchmark limitation explicit**: the IBM dataset is treated as pedagogical and non-generalizable.

## Export status

Direct file operations cannot compile binary PDF/DOCX files. The final PDF and DOCX must be generated from the prepared LaTeX source using a local document-rendering tool before journal upload.

Required final files:

- **`FINAL_POLISHED_MANUSCRIPT.pdf`**
- **`FINAL_POLISHED_MANUSCRIPT.docx`**

Do not submit placeholder export files.
