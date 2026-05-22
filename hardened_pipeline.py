"""
Hardened Evaluation Pipeline — Reviewer-Response Version
Addresses all Tier-1 and Tier-2 statistical deficiencies identified in peer review:
  - Stratified 5-Fold Cross-Validation (replaces single train/test split)
  - Expected Calibration Error (ECE) with 10 bins
  - Demographic Parity Difference (DPD) and Equalized Odds Difference (EOD)
  - PCA-projected KMeans clustering (targets silhouette > 0.30)
  - Proxy-variable mutual information audit
Outputs structured JSON for manuscript reference.
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, average_precision_score,
    accuracy_score, silhouette_score, confusion_matrix
)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_classif
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
DATA = Path("data/ibm_hr_attrition.csv")
OUT  = BASE / "hardened_results.json"

# ── helpers ───────────────────────────────────────────────────────────────────────────

def ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error (uniform binning)."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc  = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece_val += mask.sum() * abs(acc - conf)
    return float(ece_val / len(y_true))


def demographic_parity_difference(y_pred, protected):
    """DPD = P(Ŷ=1 | A=1) − P(Ŷ=1 | A=0)."""
    groups = np.unique(protected)
    rates  = {g: y_pred[protected == g].mean() for g in groups}
    vals   = list(rates.values())
    return float(max(vals) - min(vals)), rates


def equalized_odds_difference(y_true, y_pred, protected):
    """EOD = max over outcomes of |TPR_g1 − TPR_g0|, same for FPR."""
    groups = np.unique(protected)
    result = {}
    for outcome_label, outcome in [("TPR", 1), ("FPR", 0)]:
        rates = {}
        for g in groups:
            mask = protected == g
            cm   = confusion_matrix(y_true[mask], y_pred[mask], labels=[0, 1])
            if outcome == 1:     # TPR = TP / (TP + FN)
                denom = cm[1, 0] + cm[1, 1]
                rates[g] = cm[1, 1] / denom if denom > 0 else 0.0
            else:                # FPR = FP / (FP + TN)
                denom = cm[0, 0] + cm[0, 1]
                rates[g] = cm[0, 1] / denom if denom > 0 else 0.0
        vals = list(rates.values())
        result[outcome_label] = {
            "max_diff": float(max(vals) - min(vals)),
            "by_group": {str(k): float(v) for k, v in rates.items()}
        }
    eod = max(result["TPR"]["max_diff"], result["FPR"]["max_diff"])
    return float(eod), result


# ── load & prepare data ───────────────────────────────────────────────────────────────────────
print("Loading IBM HR Attrition dataset …")
df = pd.read_csv(DATA, sep=";") if DATA.suffix == ".csv" else pd.read_csv(DATA)
# detect separator if needed
if df.shape[1] == 1:
    df = pd.read_csv(DATA, sep=";")
print(f"  Shape: {df.shape}")

TARGET = "Attrition"
PROTECTED = ["Gender", "Age", "MaritalStatus"]
_candidate_drops = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours",
                    "Attrition"] + PROTECTED
DROP_COLS = [c for c in _candidate_drops if c in df.columns]

y = LabelEncoder().fit_transform(df[TARGET])

# Build structural indices
df["OvertimeFrequency"]           = (df["OverTime"] == "Yes").astype(int)
df["SatisfactionJobLevel"]        = df["JobSatisfaction"] * df["JobLevel"]
df["TenurePromotionRatio"]        = df["YearsAtCompany"] / (df["YearsSinceLastPromotion"] + 1)
df["OvertimeBalanceRatio"]        = df["OvertimeFrequency"] / (df["WorkLifeBalance"] + 1)
df["CompensationProgression"]     = df["MonthlyIncome"] / (df["YearsAtCompany"] + 1)
df["PerfCompAlignment"]           = df["PerformanceRating"] * df["MonthlyIncome"]
df["WorkArrangement"]             = df["BusinessTravel"].map(
    {"Non-Travel": 0, "Travel_Rarely": 1, "Travel_Frequently": 2}).fillna(0)
df["MultiSatisfaction"]           = (df["JobSatisfaction"] +
                                     df["EnvironmentSatisfaction"] +
                                     df["RelationshipSatisfaction"]) / 3

# Encode protected attributes separately for fairness audit
gender_enc       = (df["Gender"] == "Male").astype(int).values
marital_enc      = LabelEncoder().fit_transform(df["MaritalStatus"])
age_band_enc     = pd.cut(df["Age"], bins=[17,29,39,49,120],
                          labels=[0,1,2,3]).astype(int).values

# Feature matrix
df_dropped = df.drop(columns=DROP_COLS)
cat_cols   = df_dropped.select_dtypes("object").columns.tolist()
df_enc     = pd.get_dummies(df_dropped, columns=cat_cols)
df_enc   = df_enc.select_dtypes(include=[np.number]).fillna(0)

X = df_enc.values
feature_names = df_enc.columns.tolist()

# ── 5-Fold Stratified CV ────────────────────────────────────────────────────────────────────────
print("\nRunning Stratified 5-Fold Cross-Validation …")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_metrics = []
all_y_true, all_y_prob, all_y_pred = [], [], []
all_gender, all_marital, all_age = [], [], []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_te   = scaler.transform(X_te)

    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=0,
        class_weights={0: 1, 1: 5}
    )
    model.fit(X_tr, y_tr)

    y_prob = model.predict_proba(X_te)[:, 1]
    y_pred = (y_prob >= 0.50).astype(int)

    auc  = roc_auc_score(y_te, y_prob)
    bs   = brier_score_loss(y_te, y_prob)
    ap   = average_precision_score(y_te, y_prob)
    acc  = accuracy_score(y_te, y_pred)
    ece_ = ece(y_te, y_prob)

    dpd, dpd_detail = demographic_parity_difference(y_pred, gender_enc[test_idx])
    eod, eod_detail = equalized_odds_difference(y_te, y_pred, gender_enc[test_idx])

    fold_metrics.append({
        "fold": fold, "auc": auc, "brier": bs, "avg_prec": ap,
        "accuracy": acc, "ece": ece_, "dpd_gender": dpd, "eod_gender": eod
    })

    all_y_true.extend(y_te.tolist())
    all_y_prob.extend(y_prob.tolist())
    all_y_pred.extend(y_pred.tolist())
    all_gender.extend(gender_enc[test_idx].tolist())
    all_marital.extend(marital_enc[test_idx].tolist())
    all_age.extend(age_band_enc[test_idx].tolist())

    print(f"  Fold {fold}: AUC={auc:.4f}  Brier={bs:.4f}  ECE={ece_:.4f}"
          f"  DPD={dpd:.4f}  EOD={eod:.4f}")

all_y_true  = np.array(all_y_true)
all_y_prob  = np.array(all_y_prob)
all_y_pred  = np.array(all_y_pred)
all_gender  = np.array(all_gender)
all_marital = np.array(all_marital)
all_age     = np.array(all_age)

def ci95(vals):
    m = np.mean(vals)
    s = np.std(vals, ddof=1)
    n = len(vals)
    return m, s, m - 1.96*s/np.sqrt(n), m + 1.96*s/np.sqrt(n)

metrics_arr = {k: [f[k] for f in fold_metrics]
               for k in ["auc","brier","avg_prec","accuracy","ece","dpd_gender","eod_gender"]}

print("\n── Cross-Validation Summary ──")
cv_summary = {}
for k, vals in metrics_arr.items():
    m, s, lo, hi = ci95(vals)
    cv_summary[k] = {"mean": round(m,4), "std": round(s,4),
                     "ci95_lo": round(lo,4), "ci95_hi": round(hi,4)}
    print(f"  {k:20s}: {m:.4f} ± {s:.4f}  (95%CI [{lo:.4f}, {hi:.4f}])")

# ── Global ECE (all folds pooled) ────────────────────────────────────────────────────────────────────────
global_ece = ece(all_y_true, all_y_prob)
print(f"\nPooled ECE (10 bins): {global_ece:.4f}")

# ── Marital status fairness ─────────────────────────────────────────────────────────────────────────
dpd_marital, dpd_m_detail = demographic_parity_difference(all_y_pred, all_marital)
eod_marital, eod_m_detail = equalized_odds_difference(all_y_true, all_y_pred, all_marital)
dpd_age,     dpd_a_detail = demographic_parity_difference(all_y_pred, all_age)
eod_age,     eod_a_detail = equalized_odds_difference(all_y_true, all_y_pred, all_age)

print(f"\nFairness (pooled):")
print(f"  DPD Gender:  {np.mean(metrics_arr['dpd_gender']):.4f}")
print(f"  EOD Gender:  {np.mean(metrics_arr['eod_gender']):.4f}")
print(f"  DPD Marital: {dpd_marital:.4f}")
print(f"  EOD Marital: {eod_marital:.4f}")
print(f"  DPD Age:     {dpd_age:.4f}")
print(f"  EOD Age:     {eod_age:.4f}")

# ── Proxy-Variable Mutual Information Audit ───────────────────────────────────────────────────────
print("\nProxy-variable mutual information audit …")
protected_targets = {
    "Gender_binary":      gender_enc,
    "MaritalStatus":      marital_enc,
    "AgeBand":            age_band_enc,
}

proxy_results = {}
for prot_name, prot_vals in protected_targets.items():
    mi_scores = mutual_info_classif(df_enc.values, prot_vals, random_state=42)
    top_n     = 5
    top_idx   = np.argsort(mi_scores)[::-1][:top_n]
    proxy_results[prot_name] = {
        "top_proxies": [
            {"feature": feature_names[i], "mi_score": round(float(mi_scores[i]), 4)}
            for i in top_idx
        ],
        "any_above_0.15": bool(any(mi_scores[i] > 0.15 for i in top_idx))
    }
    print(f"  {prot_name}: top proxy = '{feature_names[top_idx[0]]}' (MI={mi_scores[top_idx[0]]:.4f})")

# ── PCA + KMeans Clustering (targets silhouette > 0.30) ──────────────────────────────────────────────────
print("\nPCA-projected KMeans clustering …")
scaler_full = StandardScaler()
X_all_scaled = scaler_full.fit_transform(X)

pca  = PCA(n_components=10, random_state=42)
X_pca = pca.fit_transform(X_all_scaled)
print(f"  PCA variance explained (10 PC): {pca.explained_variance_ratio_.sum():.3f}")

best_sil = -1
best_k   = 2
best_labels = None

for k in [2, 3, 4]:
    km     = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(X_pca)
    sil    = silhouette_score(X_pca, labels)
    print(f"  KMeans k={k}: silhouette={sil:.4f}")
    if sil > best_sil:
        best_sil, best_k, best_labels = sil, k, labels

print(f"\n  Best: k={best_k}, silhouette={best_sil:.4f}")
clustering_passed = best_sil >= 0.30
print(f"  Governance threshold (>=0.30): {'PASS' if clustering_passed else 'FAIL'}")

cluster_sizes = {int(k): int((best_labels == k).sum()) for k in np.unique(best_labels)}

# ── Assemble output ─────────────────────────────────────────────────────────────────────────────────
results = {
    "dataset": {"n_samples": int(len(y)), "n_features": int(X.shape[1]),
                "base_rate": round(float(y.mean()), 4)},
    "cross_validation": {
        "protocol": "Stratified 5-Fold CV (random_state=42)",
        "n_folds": 5,
        "per_fold": fold_metrics,
        "summary": cv_summary
    },
    "calibration": {
        "pooled_ece_10bin": round(global_ece, 4),
        "brier_cv_mean":    cv_summary["brier"]["mean"],
        "brier_cv_std":     cv_summary["brier"]["std"],
    },
    "fairness_metrics": {
        "gender": {
            "dpd": round(float(np.mean(metrics_arr["dpd_gender"])), 4),
            "eod": round(float(np.mean(metrics_arr["eod_gender"])), 4)
        },
        "marital_status": {
            "dpd": round(dpd_marital, 4),
            "eod": round(eod_marital, 4)
        },
        "age_band": {
            "dpd": round(dpd_age, 4),
            "eod": round(eod_age, 4)
        }
    },
    "proxy_audit": proxy_results,
    "clustering": {
        "method": f"PCA(10 components) + KMeans(k={best_k})",
        "pca_variance_explained": round(float(pca.explained_variance_ratio_.sum()), 4),
        "best_k": best_k,
        "silhouette": round(best_sil, 4),
        "governance_threshold": 0.30,
        "threshold_passed": clustering_passed,
        "cluster_sizes": cluster_sizes
    }
}

OUT.write_text(json.dumps(results, indent=2))
print(f"\nResults written to: {OUT}")

# ── Print LaTeX-ready numbers ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("LaTeX-ready numbers for manuscript update:")
print("="*60)
auc = cv_summary["auc"]
bs  = cv_summary["brier"]
ec  = cv_summary["ece"]
print(f"AUC:   {auc['mean']:.4f} \u00b1 {auc['std']:.4f}  [{auc['ci95_lo']:.4f}, {auc['ci95_hi']:.4f}]")
print(f"Brier: {bs['mean']:.4f} \u00b1 {bs['std']:.4f}")
print(f"ECE:   {ec['mean']:.4f} \u00b1 {ec['std']:.4f}  (pooled: {global_ece:.4f})")
print(f"DPD (Gender):  {results['fairness_metrics']['gender']['dpd']:.4f}")
print(f"EOD (Gender):  {results['fairness_metrics']['gender']['eod']:.4f}")
print(f"DPD (Marital): {results['fairness_metrics']['marital_status']['dpd']:.4f}")
print(f"EOD (Marital): {results['fairness_metrics']['marital_status']['eod']:.4f}")
print(f"Clustering:    PCA+KMeans k={best_k}, silhouette={best_sil:.4f}  ({'PASS' if clustering_passed else 'FAIL ← still below 0.30'})")
for pn, pr in proxy_results.items():
    top = pr["top_proxies"][0]
    print(f"Proxy {pn}: top={top['feature']} MI={top['mi_score']:.4f}  flagged={pr['any_above_0.15']}")
