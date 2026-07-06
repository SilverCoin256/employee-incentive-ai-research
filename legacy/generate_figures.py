"""
Figure generation — revision build.

Every figure is drawn strictly from the artifacts written by
run_real_empirical_pipeline.py:
    results/empirical_run_summary.json   (all scalar metrics)
    results/arrays.npz                   (raw predictions, importances)

No hardcoded metrics, no mock distributions. Run the pipeline first.

    python run_real_empirical_pipeline.py
    python generate_figures.py

Outputs: figures/*.{png,pdf,svg} at 600 dpi.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve, auc as auc_fn

BASE = Path(__file__).parent
RES = BASE / "results"
OUT = BASE / "figures"; OUT.mkdir(exist_ok=True)

J = json.loads((RES / "empirical_run_summary.json").read_text())
A = np.load(RES / "arrays.npz", allow_pickle=True)

# palette (unchanged from prior submission for visual consistency)
SLATE, BLUE, TEAL, AMBER, RED = "#5A6A7A", "#4A90D9", "#2E9E8C", "#E8A020", "#C0392B"
MGREY, DGREY, WHITE = "#D4D8DC", "#333333", "#FFFFFF"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8.5, "axes.titlesize": 10.0, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.35,
    "figure.facecolor": WHITE, "axes.facecolor": WHITE,
    "savefig.dpi": 600, "savefig.bbox": "tight",
})


def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=600 if ext == "png" else None,
                    bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"  saved: {name}")


# ── Fig 2: segmentation silhouettes (real, one representative per family) ─────
def fig_segmentation():
    fam = J["clustering"]["families"]
    def pick(prefix):
        vals = [v for k, v in fam.items() if k.startswith(prefix) and v is not None]
        return max(vals) if vals else None
    labels, vals = [], []
    for lab, pre in [("DBSCAN", "dbscan"), ("KMeans", "kmeans_k"),
                     ("PCA(10)+\nKMeans", "pca_kmeans"), ("GMM", "gmm")]:
        v = pick(pre)
        if v is not None:
            labels.append(lab); vals.append(v)
    thr = J["clustering"].get("threshold", 0.30)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    bars = ax.bar(labels, vals, color=SLATE, width=0.52, alpha=0.88)
    ax.axhline(0.30, color=RED, lw=1.2, ls="--", label="Governance threshold (>=0.30)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.4f}",
                ha="center", va="bottom", fontsize=8, color=DGREY)
    ax.set_ylabel("Silhouette Score"); ax.set_ylim(0, max(0.48, max(vals) + 0.1))
    ax.set_title("Silhouette scores -- governance threshold 0.30", fontsize=9.5)
    ax.legend(fontsize=7.5)
    save(fig, "fig2_segmentation")


# ── Fig 3: confusion matrix (real holdout) ───────────────────────────────────
def fig_confusion():
    cm = J["holdout"]["confusion_matrix"]
    M = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]], float)
    row = M / M.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    im = ax.imshow(row, cmap="Blues", vmin=0, vmax=1)
    names = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{names[i][j]}\n{int(M[i,j])}\n({row[i,j]:.2f})",
                    ha="center", va="center", fontsize=10,
                    color="white" if row[i, j] > 0.5 else DGREY)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred. Stay", "Pred. Attrit"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual Stay", "Actual Attrit"])
    ax.set_title(f"CatBoost · All Features · Threshold = 0.50 "
                 f"(n={J['holdout']['test_size']})", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-normalised rate")
    ax.grid(False)
    save(fig, "fig3_confusion")


# ── Fig 4: calibration curve + histogram (real reliability bins & preds) ─────
def fig_calibration():
    cal = J["calibration"][J["calibration"]["primary_model"].split("_full_")[1]] \
        if J["calibration"]["primary_model"].split("_full_")[1] in J["calibration"] \
        else J["calibration"]["weighted_5x"]
    rel = cal["reliability"]
    br = cal["brier"]
    ece = cal["ece_uniform_10"]
    prob = A["pooled_prob"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.6, 5.2),
                                   gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    ax1.plot([0, 1], [0, 1], color=MGREY, lw=0.9, ls="--", label="Perfect calibration")
    ax1.plot(rel["conf"], rel["frac_pos"], "o-", color=TEAL, lw=1.8, ms=4.5,
             label=f"CatBoost (Brier={br:.4f}, ECE={ece:.3f})")
    ax1.set_ylabel("Fraction of positives"); ax1.set_ylim(-0.02, 1.02)
    ax1.set_title("Probability calibration (pooled 5-fold CV)", fontsize=9.5)
    ax1.legend(fontsize=7.5, loc="upper left")
    ax2.hist(prob, bins=np.linspace(0, 1, 21), color=BLUE, alpha=0.75)
    ax2.set_xlabel("Mean predicted probability"); ax2.set_ylabel("Count")
    save(fig, "fig4_calibration")


# ── Fig 5: permutation importance (real) ─────────────────────────────────────
def fig_permutation():
    feats = list(A["perm_features"]); vals = list(A["perm_values"]); sds = list(A["perm_std"])
    order = np.argsort(vals)[::-1][:12][::-1]
    feats = [feats[i] for i in order]; vals = [vals[i] for i in order]; sds = [sds[i] for i in order]
    fig, ax = plt.subplots(figsize=(5.8, 4.5))
    y = np.arange(len(feats))
    ax.barh(y, vals, xerr=sds, color=SLATE, alpha=0.87, height=0.62,
            error_kw=dict(ecolor=DGREY, lw=0.6))
    ax.set_yticks(y); ax.set_yticklabels(feats, fontsize=8.2)
    ax.set_xlabel("Mean decrease in ROC-AUC (permutation)")
    ax.set_title("Permutation importance -- CatBoost", fontsize=9)
    ax.text(0.98, 0.03, "Not causal estimates", transform=ax.transAxes,
            ha="right", fontsize=6.5, color=SLATE, style="italic")
    save(fig, "fig5_permutation")


# ── Fig 6: SHAP summary (real TreeSHAP) ──────────────────────────────────────
def fig_shap():
    feats = list(A["shap_features"]); vals = list(A["shap_values"])
    order = np.argsort(vals)[::-1][:12][::-1]
    feats = [feats[i] for i in order]; vals = [vals[i] for i in order]
    comp = set(J["dataset"]["comp_group"])
    cols = [RED if f in comp else SLATE for f in feats]
    fig, ax = plt.subplots(figsize=(5.8, 4.5))
    y = np.arange(len(feats))
    ax.barh(y, vals, color=cols, alpha=0.87, height=0.62)
    ax.set_yticks(y); ax.set_yticklabels(feats, fontsize=8.2)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Mean |SHAP value| -- CatBoost ({J['shap']['source']})", fontsize=8.5)
    ax.legend(handles=[mpatches.Patch(color=RED, label="Compensation-linked"),
                       mpatches.Patch(color=SLATE, label="Other")],
              fontsize=7.2, loc="lower right")
    ax.text(0.98, 0.10, "Model-inspection only", transform=ax.transAxes,
            ha="right", fontsize=6.5, color=SLATE, style="italic")
    save(fig, "fig6_shap")


# ── Fig 7: fairness diagnostics (real DPD/EOD/DI/Cramér's V + proxy MI) ───────
def fig_fairness():
    fair = J["fairness"]; proxy = J["proxy_audit"]
    names = {"gender": "Gender", "ageband": "Age band", "compband": "Comp. band",
             "dept": "Dept.", "marital": "Marital\nstatus"}
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12.0, 3.8))

    # (a) disparate-impact ratio
    di_keys = [k for k in ["gender", "marital", "compband", "ageband", "dept"] if k in fair]
    di = [fair[k]["disparate_impact"] for k in di_keys]
    ax1.barh(range(len(di_keys)), di, color=[TEAL if v >= 0.80 else RED for v in di],
             alpha=0.87, height=0.5)
    ax1.axvline(0.80, color=RED, lw=1.3, ls="--", label="0.80 threshold")
    for i, v in enumerate(di):
        ax1.text(v + 0.02, i, f"{v:.3f}", va="center", fontsize=8,
                 color=RED if v < 0.80 else TEAL, fontweight="bold")
    ax1.set_yticks(range(len(di_keys)))
    ax1.set_yticklabels([names[k].replace("\n", " ") for k in di_keys], fontsize=8)
    ax1.set_xlabel("Disparate-Impact Ratio")
    ax1.set_title("(a) Disparate-Impact Ratios", fontsize=9); ax1.legend(fontsize=7.5)

    # (b) Cramér's V + EOD
    ck = [k for k in ["gender", "ageband", "compband", "dept", "marital"] if k in fair]
    cvv = [fair[k]["cramers_v"] for k in ck]
    eod = [fair[k]["eod"] for k in ck]
    y = np.arange(len(ck))
    ax2.barh(y - 0.2, cvv, height=0.38, color=[RED if v > 0.20 else SLATE for v in cvv],
             alpha=0.87, label="Cramér's V")
    ax2.barh(y + 0.2, eod, height=0.38, color=[RED if v > 0.20 else AMBER for v in eod],
             alpha=0.6, label="EOD")
    ax2.axvline(0.20, color=RED, lw=1.3, ls="--", label="Escalation (0.20)")
    ax2.set_yticks(y); ax2.set_yticklabels([names[k] for k in ck], fontsize=8)
    ax2.set_xlabel("Metric value"); ax2.set_title("(b) Subgroup association / EOD", fontsize=9)
    ax2.invert_yaxis(); ax2.legend(fontsize=6.8, loc="lower right")

    # (c) proxy MI (top feature per protected attribute)
    pk = [k for k in ["gender", "marital", "ageband"] if k in proxy]
    pv = [proxy[k]["top_proxies"][0]["mi_nats"] for k in pk]
    pf = [proxy[k]["top_proxies"][0]["feature"] for k in pk]
    ax3.barh(range(len(pk)), pv, color=[RED if v > 0.10 else SLATE for v in pv],
             alpha=0.87, height=0.45)
    ax3.axvline(0.30, color=RED, lw=1.3, ls="--", label="High-MI (0.30)")
    ax3.set_yticks(range(len(pk)))
    ax3.set_yticklabels([f"{names[k].replace(chr(10),' ')}\n({f})" for k, f in zip(pk, pf)],
                        fontsize=7.2)
    for i, v in enumerate(pv):
        ax3.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8, color=RED, fontweight="bold")
    ax3.set_xlabel("Mutual information (nats, top proxy)")
    ax3.set_title("(c) Proxy-Variable MI Audit", fontsize=9)
    ax3.invert_yaxis(); ax3.legend(fontsize=7)
    fig.subplots_adjust(wspace=0.5, left=0.09, right=0.98, top=0.9, bottom=0.14)
    save(fig, "fig7_fairness")


# ── Fig 8: seed sensitivity (real) + bootstrap CI band ───────────────────────
def fig_seed():
    s = J["seed_sensitivity"]; seeds = s["seeds"]
    ci = J["bootstrap"]["ci95"]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    x = np.arange(len(seeds))
    ax.plot(x, s["catboost"], "o-", color=TEAL, lw=1.8, ms=5.5, label="CatBoost")
    ax.plot(x, s["logreg"], "s--", color=BLUE, lw=1.4, ms=4.5, label="Logistic reg.")
    ax.plot(x, s["randomforest"], "^:", color=SLATE, lw=1.4, ms=4.5, label="Random forest")
    ax.axhline(s["catboost_mean"], color=TEAL, lw=0.9, ls="-.", alpha=0.6,
               label=f"CatBoost mean = {s['catboost_mean']:.4f}")
    ax.fill_between(x, [ci[0]] * len(x), [ci[1]] * len(x), color=TEAL, alpha=0.08,
                    label=f"Bootstrap 95% CI [{ci[0]:.3f}, {ci[1]:.3f}]")
    ax.set_xticks(x); ax.set_xticklabels([f"Seed\n{sd}" for sd in seeds], fontsize=7.8)
    ax.set_ylabel("ROC-AUC")
    lo = min(min(s["catboost"]), min(s["logreg"]), min(s["randomforest"]), ci[0]) - 0.02
    hi = max(max(s["catboost"]), ci[1]) + 0.02
    ax.set_ylim(lo, hi)
    ax.set_title("ROC-AUC across seven random seeds -- Bootstrap 95% CI", fontsize=9)
    ax.legend(fontsize=7.2, ncol=2)
    save(fig, "fig8_seed")


# ── Fig 9: governance dashboard (six real panels) ────────────────────────────
def fig_dashboard():
    fig, axs = plt.subplots(2, 3, figsize=(13.5, 8.0))

    # (a) ROC — real curves
    ax = axs[0, 0]
    for key, lab, col, arr in [
        ("catboost", "CatBoost", TEAL, A["holdout_prob"]),
        ("randomforest", "Random forest", BLUE, A["holdout_prob_rf"]),
        ("logreg", "Logistic reg.", SLATE, A["holdout_prob_logreg"]),
    ]:
        fpr, tpr, _ = roc_curve(A["holdout_y"], arr)
        ax.plot(fpr, tpr, color=col, lw=1.6, label=f"{lab} (AUC={auc_fn(fpr,tpr):.3f})")
    ax.plot([0, 1], [0, 1], color=MGREY, lw=0.7, ls="--")
    ci = J["bootstrap"]["ci95"]
    ax.set_title(f"(a) ROC  ·  CatBoost 95% CI [{ci[0]:.3f}, {ci[1]:.3f}]", fontsize=8.5)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend(fontsize=6.5, loc="lower right")

    # (b) calibration (primary model)
    ax = axs[0, 1]; cal = J["calibration"]["weighted_5x"]; rel = cal["reliability"]
    ax.plot([0, 1], [0, 1], color=MGREY, lw=0.9, ls="--")
    ax.plot(rel["conf"], rel["frac_pos"], "o-", color=TEAL, lw=1.6, ms=4)
    ax.set_title(f"(b) Calibration (Brier={cal['brier']:.3f})", fontsize=8.5)
    ax.set_xlabel("Mean predicted prob."); ax.set_ylabel("Fraction positive")

    # (c) ablation / feature-group contribution
    ax = axs[0, 2]; ab = J["ablation"]
    bars = ["Full", "Comp.\nremoved", "Comp.\nonly"]
    vals = [ab["full_auc"], ab["compensation_removed_auc"], ab["compensation_only_auc"]]
    ax.bar(bars, vals, color=[TEAL, SLATE, AMBER], alpha=0.85)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_ylim(0.5, 0.9); ax.set_ylabel("ROC-AUC")
    ax.set_title(f"(c) Ablation (ΔAUC removing comp = {ab['delta_removing_compensation']:+.3f})",
                 fontsize=8.5)

    # (d) SHAP
    ax = axs[1, 0]
    feats = list(A["shap_features"])[:8][::-1]; vals = list(A["shap_values"])[:8][::-1]
    comp = set(J["dataset"]["comp_group"])
    ax.barh(range(len(feats)), vals, color=[RED if f in comp else SLATE for f in feats], alpha=0.87)
    ax.set_yticks(range(len(feats))); ax.set_yticklabels(feats, fontsize=7)
    ax.set_title("(d) Feature importance (SHAP)", fontsize=8.5); ax.set_xlabel("Mean |SHAP|")

    # (e) fairness heat
    ax = axs[1, 1]; fair = J["fairness"]
    order = ["gender", "ageband", "compband", "marital"]
    order = [k for k in order if k in fair]
    metrics = ["cramers_v", "dpd", "eod", "disparate_impact"]
    Mtx = np.array([[fair[k][m] for m in metrics] for k in order])
    im = ax.imshow(Mtx, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(metrics))); ax.set_xticklabels(["V", "DPD", "EOD", "DI"], fontsize=7)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=7)
    for i in range(len(order)):
        for j in range(len(metrics)):
            ax.text(j, i, f"{Mtx[i,j]:.2f}", ha="center", va="center", fontsize=6.5)
    ax.set_title("(e) Fairness matrix", fontsize=8.5); ax.grid(False)

    # (f) seed robustness
    ax = axs[1, 2]; s = J["seed_sensitivity"]; x = np.arange(len(s["seeds"]))
    ax.plot(x, s["catboost"], "o-", color=TEAL, ms=4, label="CatBoost")
    ax.plot(x, s["randomforest"], "s--", color=BLUE, ms=3, label="RF")
    ax.plot(x, s["logreg"], "^:", color=SLATE, ms=3, label="LogReg")
    ci = J["bootstrap"]["ci95"]
    ax.fill_between(x, [ci[0]] * len(x), [ci[1]] * len(x), color=TEAL, alpha=0.08)
    ax.set_xticks(x); ax.set_xticklabels(s["seeds"], fontsize=6, rotation=45)
    ax.set_title("(f) Seed-sensitivity robustness", fontsize=8.5)
    ax.set_ylabel("ROC-AUC"); ax.legend(fontsize=6.5)

    fig.tight_layout(pad=1.2)
    save(fig, "fig9_dashboard")


if __name__ == "__main__":
    print("Generating figures from results/ …")
    for fn in (fig_segmentation, fig_confusion, fig_calibration, fig_permutation,
               fig_shap, fig_fairness, fig_seed, fig_dashboard):
        try:
            fn()
        except Exception as e:
            print(f"  !! {fn.__name__} failed: {e}")
    print(f"Done -> {OUT}/")
