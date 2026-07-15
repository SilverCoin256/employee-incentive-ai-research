"""
Single reproducible pipeline for the HR-attrition governance-audit manuscript.
Produces EVERY reported number and figure from one script/one dataset/one model version.
Run from the repository root:
    python3 pipeline.py
Reads  data/ibm_hr_attrition.csv
Writes results/results.json and figures/*.{png,pdf,svg}

Every number in the manuscript, response letter, and supplementary material is a
key in results/results.json; see TRACEABILITY.md for the number-by-number map.
"""
import json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             accuracy_score, confusion_matrix, silhouette_score)
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from scipy.stats import spearmanr
from catboost import CatBoostClassifier, Pool
import os, hashlib, sys, platform, datetime
FIG = "figures"; os.makedirs(FIG, exist_ok=True); os.makedirs("results", exist_ok=True)
SEED = 42
rng = np.random.RandomState(SEED)
R = {}  # results dict

# ---------- load ----------
DATA = "data/ibm_hr_attrition.csv"
df = pd.read_csv(DATA, encoding="utf-8-sig")
df = df.drop(columns=["EmployeeCount","Over18","StandardHours","EmployeeNumber"])
y = (df["Attrition"] == "Yes").astype(int).values
df = df.drop(columns=["Attrition"])
R["n"] = int(len(df)); R["base_rate"] = float(y.mean())

# ---------- protected attrs (excluded from model, kept for fairness) ----------
PROTECTED = ["Gender","Age","MaritalStatus"]
protected_df = df[PROTECTED].copy()

# ---------- 8 structural indices (arithmetic) ----------
def add_indices(X):
    X = X.copy()
    X["idx_OvertimeFreq"]      = (X["OverTime"]=="Yes").astype(float)
    X["idx_Satis_JobLevel"]    = X["JobSatisfaction"] / X["JobLevel"]
    X["idx_Tenure2Promo"]      = X["YearsAtCompany"] / (X["YearsSinceLastPromotion"]+1)
    X["idx_OT_Balance"]        = (X["OverTime"]=="Yes").astype(float) / X["WorkLifeBalance"]
    X["idx_CompProgression"]   = X["PercentSalaryHike"] / (X["YearsAtCompany"]+1)
    X["idx_Perf_Comp"]         = X["PerformanceRating"] * X["PercentSalaryHike"]
    X["idx_WorkArrangement"]   = X["DistanceFromHome"] * (X["BusinessTravel"]!="Non-Travel").astype(float)
    X["idx_MultiSatis"]        = X[["JobSatisfaction","EnvironmentSatisfaction",
                                    "RelationshipSatisfaction","WorkLifeBalance"]].mean(axis=1)
    return X

X_all = add_indices(df).drop(columns=PROTECTED)   # model feature space (protected excluded)

COMP = ["MonthlyIncome","DailyRate","HourlyRate","MonthlyRate","PercentSalaryHike",
        "StockOptionLevel","idx_CompProgression","idx_Perf_Comp"]

cat_cols = [c for c in X_all.columns if not pd.api.types.is_numeric_dtype(X_all[c])]
num_cols = [c for c in X_all.columns if pd.api.types.is_numeric_dtype(X_all[c])]

# ---------- helpers ----------
def cb_model(seed=SEED):
    return CatBoostClassifier(iterations=400, learning_rate=0.05, depth=6,
                              class_weights=[1,5], random_seed=seed, verbose=0,
                              allow_writing_files=False)

def ece_score(y_true, p, n_bins=10, adaptive=False):
    y_true=np.asarray(y_true); p=np.asarray(p)
    if adaptive:
        edges=np.quantile(p, np.linspace(0,1,n_bins+1)); edges[0]=-1e-9; edges[-1]=1+1e-9
    else:
        edges=np.linspace(0,1,n_bins+1)
    ece=0.0; N=len(p)
    for i in range(n_bins):
        m=(p>edges[i])&(p<=edges[i+1])
        if m.sum()==0: continue
        ece += (m.sum()/N)*abs(y_true[m].mean()-p[m].mean())
    return float(ece)

def catboost_cv_oof(Xdf, y, cols, seed=SEED):
    """Stratified 5-fold; return oof preds + per-fold aucs. Native categorical handling."""
    Xd = Xdf[cols].reset_index(drop=True)
    cats=[c for c in cols if not pd.api.types.is_numeric_dtype(Xd[c])]
    cat_idx=[Xd.columns.get_loc(c) for c in cats]
    skf=StratifiedKFold(5, shuffle=True, random_state=seed)
    oof=np.zeros(len(y)); aucs=[]; briers=[]; aps=[]; accs=[]
    for tr,va in skf.split(Xd,y):
        m=cb_model(seed)
        m.fit(Pool(Xd.iloc[tr],y[tr],cat_features=cat_idx))
        p=m.predict_proba(Pool(Xd.iloc[va],cat_features=cat_idx))[:,1]
        oof[va]=p
        aucs.append(roc_auc_score(y[va],p)); briers.append(brier_score_loss(y[va],p))
        aps.append(average_precision_score(y[va],p)); accs.append(accuracy_score(y[va],(p>=.5).astype(int)))
    return oof, np.array(aucs), np.array(briers), np.array(aps), np.array(accs), cat_idx

def ci95(a): return [float(np.percentile(a,2.5)), float(np.percentile(a,97.5))]

# ============ PRIMARY CatBoost 5-fold CV (all features) ============
oof, aucs, briers, aps, accs, cat_idx = catboost_cv_oof(X_all, y, list(X_all.columns))
R["cv"] = dict(
    auc_mean=float(aucs.mean()), auc_sd=float(aucs.std(ddof=1)),
    auc_ci=[float(aucs.mean()-1.96*aucs.std(ddof=1)/np.sqrt(5)),
            float(aucs.mean()+1.96*aucs.std(ddof=1)/np.sqrt(5))],
    brier_mean=float(briers.mean()), brier_sd=float(briers.std(ddof=1)),
    brier_ci=[float(briers.mean()-1.96*briers.std(ddof=1)/np.sqrt(5)),
              float(briers.mean()+1.96*briers.std(ddof=1)/np.sqrt(5))],
    ap_mean=float(aps.mean()), ap_sd=float(aps.std(ddof=1)),
    acc_mean=float(accs.mean()), acc_sd=float(accs.std(ddof=1)),
)
# ECE on pooled oof + bootstrap CI
R["ece_10bin"]=ece_score(y,oof,10,False)
R["ece_adaptive"]=ece_score(y,oof,10,True)
boot_ece=[]
idx=np.arange(len(y))
for _ in range(1000):
    b=rng.choice(idx,len(idx),replace=True)
    boot_ece.append(ece_score(y[b],oof[b],10,False))
R["ece_ci"]=ci95(np.array(boot_ece))
R["mean_pred_weighted"]=float(oof.mean())

# unweighted mean pred (no class weight) for the 5:1 statement
m_unw=CatBoostClassifier(iterations=400,learning_rate=0.05,depth=6,random_seed=SEED,verbose=0,allow_writing_files=False)
skf=StratifiedKFold(5,shuffle=True,random_state=SEED); oof_unw=np.zeros(len(y))
Xd=X_all.reset_index(drop=True)
for tr,va in skf.split(Xd,y):
    mm=CatBoostClassifier(iterations=400,learning_rate=0.05,depth=6,random_seed=SEED,verbose=0,allow_writing_files=False)
    mm.fit(Pool(Xd.iloc[tr],y[tr],cat_features=cat_idx))
    oof_unw[va]=mm.predict_proba(Pool(Xd.iloc[va],cat_features=cat_idx))[:,1]
R["mean_pred_unweighted"]=float(oof_unw.mean())

# ============ Holdout 80/20 ============
Xtr,Xte,ytr,yte=train_test_split(X_all,y,test_size=0.2,stratify=y,random_state=SEED)
mh=cb_model(); mh.fit(Pool(Xtr,ytr,cat_features=cat_idx))
p_te=mh.predict_proba(Pool(Xte,cat_features=cat_idx))[:,1]
R["holdout_auc"]=float(roc_auc_score(yte,p_te))
R["holdout_brier"]=float(brier_score_loss(yte,p_te))
# training AUC (optimism)
p_tr=mh.predict_proba(Pool(Xtr,cat_features=cat_idx))[:,1]
R["train_auc"]=float(roc_auc_score(ytr,p_tr))

# bootstrap CI on holdout AUC
boot_auc=[]
ii=np.arange(len(yte))
for _ in range(1000):
    b=rng.choice(ii,len(ii),replace=True)
    if len(np.unique(yte[b]))<2: continue
    boot_auc.append(roc_auc_score(yte[b],p_te[b]))
R["bootstrap_auc_ci"]=ci95(np.array(boot_auc))

# ============ Benchmark matrix (Table 3) ============
def summarize(a,b,ap,ac,oof_):
    """Per-fold arrays + pooled OOF -> mean, fold-SD, normal-approx 95% CI, pooled ECE."""
    return dict(auc=float(a.mean()),auc_sd=float(a.std(ddof=1)),
                auc_ci=[float(a.mean()-1.96*a.std(ddof=1)/np.sqrt(5)),
                        float(a.mean()+1.96*a.std(ddof=1)/np.sqrt(5))],
                ap=float(ap.mean()),ap_sd=float(ap.std(ddof=1)),
                brier=float(b.mean()),brier_sd=float(b.std(ddof=1)),
                acc=float(ac.mean()),acc_sd=float(ac.std(ddof=1)),
                ece=ece_score(y,oof_,10,False))
def eval_cb(cols):
    oof_,a,b,ap,ac,_=catboost_cv_oof(X_all,y,cols)
    return summarize(a,b,ap,ac,oof_)
comp_removed=[c for c in X_all.columns if c not in COMP]
R["bench"]={}
R["bench"]["cb_all"]=summarize(aucs,briers,aps,accs,oof)
R["bench"]["cb_comp_removed"]=eval_cb(comp_removed)
R["bench"]["cb_comp_only"]=eval_cb(COMP)

# CatBoost without class weighting: per-fold metrics from oof_unw (same folds, no refit)
a=[];b=[];ap_=[];ac=[]
for tr,va in StratifiedKFold(5,shuffle=True,random_state=SEED).split(Xd,y):
    p=oof_unw[va]
    a.append(roc_auc_score(y[va],p));b.append(brier_score_loss(y[va],p))
    ap_.append(average_precision_score(y[va],p));ac.append(accuracy_score(y[va],(p>=.5).astype(int)))
R["bench"]["cb_all_unweighted"]=summarize(np.array(a),np.array(b),np.array(ap_),np.array(ac),oof_unw)

# LR & RF need numeric encoding
pre=ColumnTransformer([("num",StandardScaler(),num_cols),
                       ("cat",OneHotEncoder(handle_unknown="ignore"),cat_cols)])
def eval_sklearn(est):
    skf=StratifiedKFold(5,shuffle=True,random_state=SEED)
    a=[];ap=[];br=[];ac=[];oof_=np.zeros(len(y))
    for tr,va in skf.split(X_all,y):
        pipe=Pipeline([("pre",pre),("clf",est)])
        pipe.fit(X_all.iloc[tr],y[tr])
        p=pipe.predict_proba(X_all.iloc[va])[:,1]
        oof_[va]=p
        a.append(roc_auc_score(y[va],p));ap.append(average_precision_score(y[va],p))
        br.append(brier_score_loss(y[va],p));ac.append(accuracy_score(y[va],(p>=.5).astype(int)))
    return summarize(np.array(a),np.array(br),np.array(ap),np.array(ac),oof_), oof_
R["bench"]["lr"],oof_lr_bal=eval_sklearn(LogisticRegression(max_iter=1000,C=1.0,class_weight="balanced"))
R["bench"]["lr_unweighted"],oof_lr=eval_sklearn(LogisticRegression(max_iter=1000,C=1.0))
R["bench"]["rf"],_=eval_sklearn(RandomForestClassifier(n_estimators=400,min_samples_leaf=2,random_state=SEED,class_weight="balanced"))
R["bench"]["rf_unweighted"],_=eval_sklearn(RandomForestClassifier(n_estimators=400,min_samples_leaf=2,random_state=SEED))
# majority
skf=StratifiedKFold(5,shuffle=True,random_state=SEED); a=[];ap=[];br=[];ac=[]
for tr,va in skf.split(X_all,y):
    dc=DummyClassifier(strategy="prior").fit(X_all.iloc[tr],y[tr])
    p=dc.predict_proba(X_all.iloc[va])[:,1]
    a.append(roc_auc_score(y[va],p) if len(np.unique(y[va]))>1 else 0.5)
    ap.append(average_precision_score(y[va],p));br.append(brier_score_loss(y[va],p))
    ac.append(accuracy_score(y[va],(p>=.5).astype(int)))
R["bench"]["dummy"]=dict(auc=0.5,ap=float(np.mean(ap)),brier=float(np.mean(br)),acc=float(np.mean(ac)))

# ============ Seed sensitivity (7 seeds) ============
seed_aucs=[]
for s in [0,1,7,13,21,42,101]:
    Xtr2,Xte2,ytr2,yte2=train_test_split(X_all,y,test_size=0.2,stratify=y,random_state=s)
    mm=cb_model(s); mm.fit(Pool(Xtr2,ytr2,cat_features=cat_idx))
    pp=mm.predict_proba(Pool(Xte2,cat_features=cat_idx))[:,1]
    seed_aucs.append(float(roc_auc_score(yte2,pp)))
R["seed_aucs"]=seed_aucs; R["seed_min"]=min(seed_aucs); R["seed_max"]=max(seed_aucs)

# ============ Fairness (pooled 5-fold oof, thr 0.5) ============
def cramers_v(a,b):
    ct=pd.crosstab(a,b); chi2=0.0
    from scipy.stats import chi2_contingency
    chi2=chi2_contingency(ct)[0]
    n=ct.values.sum(); r,k=ct.shape
    return float(np.sqrt((chi2/n)/max(1,(min(r-1,k-1)))))
def fairness(group):
    pred=(oof>=.5).astype(int)
    g=pd.Series(group).reset_index(drop=True)
    rates={}; tprs={}; fprs={}
    for lvl in g.unique():
        m=(g==lvl).values
        rates[lvl]=pred[m].mean()
        pos=m&(y==1); neg=m&(y==0)
        tprs[lvl]=pred[pos].mean() if pos.sum()>0 else np.nan
        fprs[lvl]=pred[neg].mean() if neg.sum()>0 else np.nan
    rv=np.array(list(rates.values()))
    dir_=float(rv.min()/rv.max()) if rv.max()>0 else np.nan
    dpd=float(rv.max()-rv.min())
    tt=np.array([v for v in tprs.values() if not np.isnan(v)])
    ff=np.array([v for v in fprs.values() if not np.isnan(v)])
    eod=float(max(tt.max()-tt.min(), ff.max()-ff.min()))
    cv=cramers_v(pd.Series(pred), g)
    return dict(dir=dir_,cv=cv,dpd=dpd,eod=eod)
age_band=pd.cut(protected_df["Age"],[17,30,40,50,100],labels=["<=30","31-40","41-50","50+"])
comp_band=pd.qcut(df["MonthlyIncome"],4,labels=["Q1","Q2","Q3","Q4"])
R["fair"]={
 "Gender":fairness(protected_df["Gender"].values),
 "Age band":fairness(age_band.values),
 "Compensation band":fairness(comp_band.values),
 "Marital status":fairness(protected_df["MaritalStatus"].values),
 "Department":fairness(df["Department"].values),
}

# ============ Proxy MI audit ============
# encode all model features numerically for MI
Xmi=X_all.copy()
for c in cat_cols: Xmi[c]=Xmi[c].astype("category").cat.codes
disc=[Xmi.columns.get_loc(c) for c in cat_cols]
def top_mi_classif(target):
    mi=mutual_info_classif(Xmi.values,target,discrete_features=disc,random_state=SEED)
    s=pd.Series(mi,index=Xmi.columns).sort_values(ascending=False)
    return s
def top_mi_reg(target):
    mi=mutual_info_regression(Xmi.values,target,discrete_features=disc,random_state=SEED)
    s=pd.Series(mi,index=Xmi.columns).sort_values(ascending=False)
    return s
mi_gender=top_mi_classif((protected_df["Gender"]=="Male").astype(int).values)
mi_marital=top_mi_classif(protected_df["MaritalStatus"].astype("category").cat.codes.values)
mi_age=top_mi_reg(protected_df["Age"].values.astype(float))
R["mi"]={
 "Gender":{"top":mi_gender.index[0],"val":float(mi_gender.iloc[0])},
 "MaritalStatus":{"top":mi_marital.index[0],"val":float(mi_marital.iloc[0]),
                  "top3":[(k,round(float(v),3)) for k,v in mi_marital.head(3).items()]},
 "Age":{"top":mi_age.index[0],"val":float(mi_age.iloc[0]),
        "top3":[(k,round(float(v),3)) for k,v in mi_age.head(3).items()]},
}

# ============ Segmentation ============
Xseg=StandardScaler().fit_transform(X_all[num_cols].fillna(0))
seg={}
km=KMeans(2,random_state=SEED,n_init=10).fit(Xseg); seg["KMeans"]=float(silhouette_score(Xseg,km.labels_))
pca=PCA(10,random_state=SEED).fit(Xseg); Xp=pca.transform(Xseg)
R["pca_var10"]=float(pca.explained_variance_ratio_[:10].sum())
kmp=KMeans(2,random_state=SEED,n_init=10).fit(Xp); seg["PCA10+KMeans"]=float(silhouette_score(Xp,kmp.labels_))
gm=GaussianMixture(2,random_state=SEED).fit(Xseg); seg["GMM"]=float(silhouette_score(Xseg,gm.predict(Xseg)))
dbscan_ok=False
for eps in np.arange(1.5,3.01,0.25):
    lab=DBSCAN(eps=eps,min_samples=10).fit(Xseg).labels_
    if len(set(lab)-{-1})>=2:
        dbscan_ok=True; break
seg["DBSCAN_ge2"]=dbscan_ok
R["seg"]=seg

# ============ Permutation importance + SHAP ============
def auc_of(model,Xf): return roc_auc_score(yte, model.predict_proba(Pool(Xf,cat_features=cat_idx))[:,1])
base_auc=auc_of(mh,Xte)
perm_imp={}
for col in X_all.columns:
    drops=[]
    for _ in range(10):
        Xperm=Xte.copy(); Xperm[col]=rng.permutation(Xperm[col].values)
        drops.append(base_auc-auc_of(mh,Xperm))
    perm_imp[col]=float(np.mean(drops))
perm_s=pd.Series(perm_imp).sort_values(ascending=False)
R["perm_top"]=[(k,round(float(v),4)) for k,v in perm_s.head(8).items()]

# SHAP
expl=mh.get_feature_importance(Pool(Xte,yte,cat_features=cat_idx),type="ShapValues")
shap_vals=expl[:,:-1]
shap_mean=np.abs(shap_vals).mean(0)
shap_s=pd.Series(shap_mean,index=X_all.columns).sort_values(ascending=False)
R["shap_top"]=[(k,round(float(v),4)) for k,v in shap_s.head(8).items()]
# spearman between shap-rank and perm-rank
common=list(X_all.columns)
rho=spearmanr([shap_s[c] for c in common],[perm_s[c] for c in common]).correlation
R["shap_perm_spearman"]=float(rho)
# sanity: permuted-label model perm importance ~ 0
yperm=rng.permutation(ytr)
msan=cb_model(); msan.fit(Pool(Xtr,yperm,cat_features=cat_idx))
base_s=roc_auc_score(yte,msan.predict_proba(Pool(Xte,cat_features=cat_idx))[:,1])
sane=[]
for col in shap_s.head(5).index:
    Xperm=Xte.copy(); Xperm[col]=rng.permutation(Xperm[col].values)
    sane.append(abs(base_s-roc_auc_score(yte,msan.predict_proba(Pool(Xperm,cat_features=cat_idx))[:,1])))
R["sanity_permlabel_maxdrop"]=float(np.max(sane))

# =====================================================================
# EXTENDED AUDIT COMPUTATIONS (second revision)
# Every block below uses its OWN RandomState; nothing reuses the global
# `rng`, so all results above reproduce byte-identically.
# =====================================================================

# ---------- E1. Per-model optimism gap (train vs CV vs holdout AUC) ----------
gaps={}
gaps["catboost_weighted"]=dict(train_auc=R["train_auc"],holdout_auc=R["holdout_auc"],
                               cv_auc=R["cv"]["auc_mean"])
mg=CatBoostClassifier(iterations=400,learning_rate=0.05,depth=6,random_seed=SEED,verbose=0,allow_writing_files=False)
mg.fit(Pool(Xtr,ytr,cat_features=cat_idx))
gaps["catboost_unweighted"]=dict(
    train_auc=float(roc_auc_score(ytr,mg.predict_proba(Pool(Xtr,cat_features=cat_idx))[:,1])),
    holdout_auc=float(roc_auc_score(yte,mg.predict_proba(Pool(Xte,cat_features=cat_idx))[:,1])),
    cv_auc=R["bench"]["cb_all_unweighted"]["auc"])
for name,est,key in [
        ("logreg_unweighted",LogisticRegression(max_iter=1000,C=1.0),"lr_unweighted"),
        ("logreg_balanced",LogisticRegression(max_iter=1000,C=1.0,class_weight="balanced"),"lr"),
        ("rf_balanced",RandomForestClassifier(n_estimators=400,min_samples_leaf=2,random_state=SEED,class_weight="balanced"),"rf"),
        ("rf_unweighted",RandomForestClassifier(n_estimators=400,min_samples_leaf=2,random_state=SEED),"rf_unweighted")]:
    pipe=Pipeline([("pre",pre),("clf",est)]).fit(Xtr,ytr)
    gaps[name]=dict(train_auc=float(roc_auc_score(ytr,pipe.predict_proba(Xtr)[:,1])),
                    holdout_auc=float(roc_auc_score(yte,pipe.predict_proba(Xte)[:,1])),
                    cv_auc=R["bench"][key]["auc"])
for v in gaps.values(): v["optimism_gap"]=round(v["train_auc"]-v["cv_auc"],4)
R["overfit_gaps"]=gaps

# ---------- E2. CatBoost regularization sweep (overfitting assessment) ----------
sweep=[]
for depth,iters,l2 in [(6,400,3),(4,400,3),(3,400,3),(6,100,3),(4,100,3),(6,400,10)]:
    def mk(): return CatBoostClassifier(iterations=iters,learning_rate=0.05,depth=depth,
              l2_leaf_reg=l2,class_weights=[1,5],random_seed=SEED,verbose=0,allow_writing_files=False)
    aa=[]
    for tr,va in StratifiedKFold(5,shuffle=True,random_state=SEED).split(Xd,y):
        m_=mk(); m_.fit(Pool(Xd.iloc[tr],y[tr],cat_features=cat_idx))
        aa.append(roc_auc_score(y[va],m_.predict_proba(Pool(Xd.iloc[va],cat_features=cat_idx))[:,1]))
    m_=mk(); m_.fit(Pool(Xtr,ytr,cat_features=cat_idx))
    tr_auc=float(roc_auc_score(ytr,m_.predict_proba(Pool(Xtr,cat_features=cat_idx))[:,1]))
    sweep.append(dict(depth=depth,iterations=iters,l2_leaf_reg=l2,
                      cv_auc=float(np.mean(aa)),cv_auc_sd=float(np.std(aa,ddof=1)),train_auc=tr_auc))
R["cb_regularization_sweep"]=sweep

# ---------- E3. Prespecified model-selection rule ----------
# Gate: pooled CV ECE < 0.10 AND CV Brier below the base-rate (dummy) Brier
# (positive Brier skill). Then: highest mean CV AUC; every gated config within
# one fold-SD of the best -> select the least complex family (LR < RF < CatBoost).
brier_ref=R["bench"]["dummy"]["brier"]
candidates={"lr_unweighted":1,"lr":1,"rf_unweighted":2,"rf":2,"cb_all_unweighted":3,"cb_all":3}
gated={k:R["bench"][k] for k in candidates
       if R["bench"][k]["ece"]<0.10 and R["bench"][k]["brier"]<brier_ref}
best=max(gated,key=lambda k:gated[k]["auc"])
tol=gated[best]["auc"]-gated[best]["auc_sd"]
within={k:v for k,v in gated.items() if v["auc"]>=tol}
selected=min(within,key=lambda k:(candidates[k],-within[k]["auc"]))
R["model_selection"]=dict(
    rule=("Gate: pooled CV ECE<0.10 and CV Brier < base-rate (dummy) Brier; "
          "then max CV AUC; configs within one fold-SD of the best -> least "
          "complex model family (LR < RF < CatBoost)."),
    brier_reference_dummy=float(brier_ref),
    gated=sorted(gated),failed_gate=sorted(set(candidates)-set(gated)),
    best_auc_config=best,one_sd_tolerance=float(tol),
    within_tolerance=sorted(within),selected=selected)

# ---------- E4. Repeated full-pipeline resampling (split-to-split variability) ----------
# 100 stratified 80/20 splits; encoding fits, class weighting, and model training
# are all repeated inside each split. This estimates variability across alternative
# data splits and refits; the fixed-holdout bootstrap above is conditional on one
# fitted model and one split, and is interpreted as such in the manuscript.
def full_split_metrics(rs):
    Xtr_,Xte_,ytr_,yte_=train_test_split(X_all,y,test_size=0.2,stratify=y,random_state=rs)
    m_=cb_model(SEED); m_.fit(Pool(Xtr_,ytr_,cat_features=cat_idx))
    p_=m_.predict_proba(Pool(Xte_,cat_features=cat_idx))[:,1]
    lr_=Pipeline([("pre",pre),("clf",LogisticRegression(max_iter=1000,C=1.0))]).fit(Xtr_,ytr_)
    pl_=lr_.predict_proba(Xte_)[:,1]
    return (roc_auc_score(yte_,p_),brier_score_loss(yte_,p_),ece_score(yte_,p_,10,False),
            roc_auc_score(yte_,pl_))
rep=np.array([full_split_metrics(2000+r) for r in range(100)])
R["repeated_splits"]=dict(
    n_repeats=100,
    cb_auc_mean=float(rep[:,0].mean()),cb_auc_sd=float(rep[:,0].std(ddof=1)),
    cb_auc_p2_5_97_5=ci95(rep[:,0]),
    cb_brier_mean=float(rep[:,1].mean()),cb_brier_p2_5_97_5=ci95(rep[:,1]),
    cb_ece_mean=float(rep[:,2].mean()),cb_ece_p2_5_97_5=ci95(rep[:,2]),
    lr_auc_mean=float(rep[:,3].mean()),lr_auc_sd=float(rep[:,3].std(ddof=1)),
    lr_auc_p2_5_97_5=ci95(rep[:,3]))
rep_split_aucs=rep[:,0]

# ---------- E5. XAI validation with repeated controls ----------
# (a) SHAP vs permutation-importance rank agreement across 10 permutation seeds
def perm_importance_seed(seed):
    r_=np.random.RandomState(seed); imp={}
    for col in X_all.columns:
        drops=[]
        for _ in range(10):
            Xp_=Xte.copy(); Xp_[col]=r_.permutation(Xp_[col].values)
            drops.append(base_auc-auc_of(mh,Xp_))
        imp[col]=float(np.mean(drops))
    return pd.Series(imp)
rhos=[]
for s in range(10):
    ps_=perm_importance_seed(3000+s)
    rhos.append(float(spearmanr([shap_s[c] for c in common],[ps_[c] for c in common]).correlation))
R["xai_spearman"]=dict(point=R["shap_perm_spearman"],repeats=[round(r_,4) for r_ in rhos],
                       mean=float(np.mean(rhos)),sd=float(np.std(rhos,ddof=1)),
                       min=float(np.min(rhos)),max=float(np.max(rhos)))

# (b) deletion-faithfulness: jointly permute top-5 SHAP features vs random 5-feature sets
r_mask=np.random.RandomState(4000)
top5=list(shap_s.head(5).index)
def joint_mask_drop(cols_,r_,reps):
    ds=[]
    for _ in range(reps):
        Xp_=Xte.copy()
        for c in cols_: Xp_[c]=r_.permutation(Xp_[c].values)
        ds.append(base_auc-auc_of(mh,Xp_))
    return np.array(ds)
top_drops=joint_mask_drop(top5,r_mask,20)
rand_means=[]
for _ in range(20):
    cols_=list(r_mask.choice(np.array(X_all.columns),5,replace=False))
    rand_means.append(float(joint_mask_drop(cols_,r_mask,5).mean()))
rand_means=np.array(rand_means)
R["xai_masking"]=dict(top5_features=top5,n_top_repeats=20,n_random_sets=20,
    top5_dauc_mean=float(top_drops.mean()),top5_dauc_sd=float(top_drops.std(ddof=1)),
    random5_dauc_mean=float(rand_means.mean()),random5_dauc_sd=float(rand_means.std(ddof=1)),
    z_vs_random=float((top_drops.mean()-rand_means.mean())/rand_means.std(ddof=1)))

# (c) shuffled-label control: 10 refits on permuted training labels
sl_auc=[];sl_maxdrop=[]
for i in range(10):
    r_=np.random.RandomState(5000+i)
    ms_=cb_model(SEED); ms_.fit(Pool(Xtr,r_.permutation(ytr),cat_features=cat_idx))
    ps_=ms_.predict_proba(Pool(Xte,cat_features=cat_idx))[:,1]
    sl_auc.append(float(roc_auc_score(yte,ps_)))
    b0=sl_auc[-1];dd=[]
    for col in top5:
        Xp_=Xte.copy();Xp_[col]=r_.permutation(Xp_[col].values)
        dd.append(abs(b0-roc_auc_score(yte,ms_.predict_proba(Pool(Xp_,cat_features=cat_idx))[:,1])))
    sl_maxdrop.append(float(np.max(dd)))
R["xai_shuffled_label"]=dict(n_repeats=10,
    auc_mean=float(np.mean(sl_auc)),auc_sd=float(np.std(sl_auc,ddof=1)),
    auc_min=float(np.min(sl_auc)),auc_max=float(np.max(sl_auc)),
    maxdrop_mean=float(np.mean(sl_maxdrop)),maxdrop_sd=float(np.std(sl_maxdrop,ddof=1)))

# ---------- E6. MI permutation nulls (200 permutations, top proxy feature) ----------
def mi_null(feature,target,is_reg,n_perm=200,seed=6000):
    r_=np.random.RandomState(seed)
    x=Xmi[[feature]].values
    d=[0] if feature in cat_cols else False
    f=mutual_info_regression if is_reg else mutual_info_classif
    obs=float(f(x,target,discrete_features=d,random_state=SEED)[0])
    null=np.array([float(f(x,r_.permutation(target),discrete_features=d,random_state=SEED)[0])
                   for _ in range(n_perm)])
    return dict(feature=feature,observed=obs,null_mean=float(null.mean()),
                null_sd=float(null.std(ddof=1)),
                z=float((obs-null.mean())/max(null.std(ddof=1),1e-12)),
                p_perm=float((np.sum(null>=obs)+1)/(n_perm+1)),n_perm=n_perm)
R["mi_null"]={
 "Gender":mi_null(mi_gender.index[0],(protected_df["Gender"]=="Male").astype(int).values,False,seed=6000),
 "MaritalStatus":mi_null(mi_marital.index[0],protected_df["MaritalStatus"].astype("category").cat.codes.values,False,seed=6001),
 "Age":mi_null(mi_age.index[0],protected_df["Age"].values.astype(float),True,seed=6002)}

# ---------- E7. Calibration extras (unweighted ECE with CI; holdout ECE) ----------
R["ece_10bin_unweighted"]=ece_score(y,oof_unw,10,False)
r_e=np.random.RandomState(8000); be=[]
for _ in range(1000):
    b=r_e.choice(idx,len(idx),replace=True)
    be.append(ece_score(y[b],oof_unw[b],10,False))
R["ece_unweighted_ci"]=ci95(np.array(be))
R["holdout_ece"]=ece_score(yte,p_te,10,False)

# ---------- E8. Fairness: subgroup sizes + bootstrap CIs; LR comparison ----------
def fairness_full(group,preds,n_boot,seed):
    g=pd.Series(group).reset_index(drop=True).astype(str).values
    pred=(preds>=.5).astype(int)
    levels=sorted(pd.unique(g))
    sizes={l:dict(n=int((g==l).sum()),n_pos=int(((g==l)&(y==1)).sum())) for l in levels}
    def metrics(bidx):
        gb=g[bidx];pb=pred[bidx];yb=y[bidx]
        rates=[];tprs=[];fprs=[]
        for l in levels:
            m=gb==l
            if m.sum()==0: return None
            rates.append(pb[m].mean())
            pos=m&(yb==1);neg=m&(yb==0)
            if pos.sum()>0: tprs.append(pb[pos].mean())
            if neg.sum()>0: fprs.append(pb[neg].mean())
        rates=np.array(rates);tprs=np.array(tprs);fprs=np.array(fprs)
        dir_=rates.min()/rates.max() if rates.max()>0 else np.nan
        dpd=rates.max()-rates.min()
        eod=max(tprs.max()-tprs.min() if len(tprs) else 0.0,
                fprs.max()-fprs.min() if len(fprs) else 0.0)
        v=cramers_v(pd.Series(pb),pd.Series(gb))
        return dir_,dpd,eod,v
    point=metrics(np.arange(len(y)))
    r_=np.random.RandomState(seed);boot=[];tries=0
    while len(boot)<n_boot and tries<n_boot*3:
        tries+=1
        m_=metrics(r_.choice(len(y),len(y),replace=True))
        if m_ is not None and not any(np.isnan(np.array(m_,dtype=float))): boot.append(m_)
    boot=np.array(boot)
    return dict(dir=float(point[0]),dpd=float(point[1]),eod=float(point[2]),cv=float(point[3]),
                dir_ci=ci95(boot[:,0]),dpd_ci=ci95(boot[:,1]),
                eod_ci=ci95(boot[:,2]),cv_ci=ci95(boot[:,3]),
                n_boot_effective=int(len(boot)),groups=sizes)
FAIR_ATTRS=[("Gender",protected_df["Gender"].values),("Age band",age_band.values),
            ("Compensation band",comp_band.values),("Marital status",protected_df["MaritalStatus"].values),
            ("Department",df["Department"].values)]
R["fair_full"]={name:fairness_full(grp,oof,1000,7000+i) for i,(name,grp) in enumerate(FAIR_ATTRS)}
R["fair_lr_unweighted"]={name:fairness_full(grp,oof_lr,200,7500+i) for i,(name,grp) in enumerate(FAIR_ATTRS)}

# ---------- E9. Governance-threshold sensitivity (all thresholds) ----------
sil_max=float(max(seg["KMeans"],seg["PCA10+KMeans"],seg["GMM"]))
ts={}
ts["ece"]=dict(value_weighted=R["ece_10bin"],value_unweighted=R["ece_10bin_unweighted"],
    grid={f"{t:.2f}":bool(R["ece_10bin"]<t) for t in np.arange(0.02,0.155,0.01)})
ts["brier"]=dict(value=R["cv"]["brier_mean"],base_rate_reference=float(brier_ref),
    brier_skill_score=float(1-R["cv"]["brier_mean"]/brier_ref),
    grid={f"{t:.2f}":bool(R["cv"]["brier_mean"]<t) for t in np.arange(0.10,0.155,0.01)},
    note=("base-rate (climatological) Brier = reference forecast; the audit anchors "
          "the calibration gate at positive Brier skill rather than a fixed 0.15 cut-off"))
ts["silhouette"]=dict(value_max=sil_max,
    grid={f"{t:.2f}":bool(sil_max>=t) for t in np.arange(0.15,0.51,0.05)})
fa=R["fair"]
ts["eod"]=dict(grid={f"{t:.2f}":sorted([a for a in fa if fa[a]["eod"]>t]) for t in [0.10,0.15,0.20,0.25,0.30,0.35]})
ts["dir"]=dict(grid={f"{t:.2f}":sorted([a for a in fa if fa[a]["dir"]<t]) for t in [0.60,0.70,0.80,0.90]})
ts["cramers_v"]=dict(grid={f"{t:.2f}":sorted([a for a in fa if fa[a]["cv"]>t]) for t in [0.10,0.15,0.20,0.25,0.30]})
R["threshold_sensitivity"]=ts

# ---------- E10. Provenance ----------
import sklearn as _sk, catboost as _cb, scipy as _sp
R["provenance"]=dict(seed=SEED,dataset_file=DATA,
    dataset_sha256=hashlib.sha256(open(DATA,"rb").read()).hexdigest(),
    n_rows=int(len(y)),python=sys.version.split()[0],platform=platform.platform(),
    numpy=np.__version__,pandas=pd.__version__,sklearn=_sk.__version__,
    catboost=_cb.__version__,scipy=_sp.__version__,
    generated_utc=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))

# ================= FIGURES =================
plt.rcParams.update({"figure.dpi":150,"font.size":10})
BLUE="#2b6cb0"; RED="#c53030"; GREY="#718096"; GREEN="#2f855a"

# Fig2 segmentation
fig,ax=plt.subplots(figsize=(6,4))
names=["KMeans","PCA(10)+\nKMeans","GMM","DBSCAN"]
vals=[seg["KMeans"],seg["PCA10+KMeans"],seg["GMM"],0]
bars=ax.bar(names,vals,color=[BLUE,BLUE,BLUE,GREY])
ax.axhline(0.30,ls="--",color=RED,label="governance threshold 0.30")
for b,v,ok in zip(bars,vals,[1,1,1,0]):
    ax.text(b.get_x()+b.get_width()/2, v+0.01, (f"{v:.3f}" if ok else "no ≥2-cluster"),
            ha="center",fontsize=8)
ax.set_ylabel("Silhouette"); ax.set_ylim(0,0.4); ax.legend()
ax.set_title("Segmentation validity across algorithm families")
plt.tight_layout(); plt.savefig("figures/fig2_segmentation.png"); plt.savefig("figures/fig2_segmentation.pdf"); plt.savefig("figures/fig2_segmentation.svg"); plt.close()

# Fig3 confusion matrix (holdout)
cm=confusion_matrix(yte,(p_te>=.5).astype(int))
fig,ax=plt.subplots(figsize=(4.2,3.8))
im=ax.imshow(cm,cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j,i,cm[i,j],ha="center",va="center",
                color="white" if cm[i,j]>cm.max()/2 else "black",fontsize=13)
ax.set_xticks([0,1]); ax.set_xticklabels(["Stay","Attrit"])
ax.set_yticks([0,1]); ax.set_yticklabels(["Stay","Attrit"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix (holdout, thr=0.50)")
plt.tight_layout(); plt.savefig("figures/fig3_confusion.png"); plt.savefig("figures/fig3_confusion.pdf"); plt.savefig("figures/fig3_confusion.svg"); plt.close()
R["cm"]=cm.tolist()

# Fig4 calibration curve
from sklearn.calibration import calibration_curve
frac,mean_pred=calibration_curve(y,oof,n_bins=10,strategy="uniform")
fig,ax=plt.subplots(figsize=(5,4))
ax.plot([0,1],[0,1],"--",color=GREY,label="perfect")
ax.plot(mean_pred,frac,"o-",color=BLUE,label=f"CatBoost (ECE={R['ece_10bin']:.3f})")
ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed frequency")
ax.set_title(f"Calibration curve (Brier={R['cv']['brier_mean']:.3f})"); ax.legend()
plt.tight_layout(); plt.savefig("figures/fig4_calibration.png"); plt.savefig("figures/fig4_calibration.pdf"); plt.savefig("figures/fig4_calibration.svg"); plt.close()

# Fig5 permutation importance
fig,ax=plt.subplots(figsize=(6,4))
top=perm_s.head(10)[::-1]
ax.barh(range(len(top)),top.values,color=BLUE)
ax.set_yticks(range(len(top))); ax.set_yticklabels(top.index,fontsize=8)
ax.set_xlabel("Mean AUC drop"); ax.set_title("Permutation importance (holdout)")
plt.tight_layout(); plt.savefig("figures/fig5_permutation.png"); plt.savefig("figures/fig5_permutation.pdf"); plt.savefig("figures/fig5_permutation.svg"); plt.close()

# Fig6 SHAP summary (bar of mean|SHAP|)
fig,ax=plt.subplots(figsize=(6,4))
tops=shap_s.head(10)[::-1]
ax.barh(range(len(tops)),tops.values,color=GREEN)
ax.set_yticks(range(len(tops))); ax.set_yticklabels(tops.index,fontsize=8)
ax.set_xlabel("mean |SHAP value|"); ax.set_title("SHAP feature importance (holdout)")
plt.tight_layout(); plt.savefig("figures/fig6_shap.png"); plt.savefig("figures/fig6_shap.pdf"); plt.savefig("figures/fig6_shap.svg"); plt.close()

# Fig7 fairness (3-panel)
fig,axes=plt.subplots(1,3,figsize=(13,4))
attrs=list(R["fair"].keys())
dirs=[R["fair"][a]["dir"] for a in attrs]
axes[0].bar(attrs,dirs,color=BLUE); axes[0].axhline(0.8,ls="--",color=RED)
axes[0].set_title("Disparate-impact ratio"); axes[0].tick_params(axis="x",rotation=30)
cvs=[R["fair"][a]["cv"] for a in attrs]; eods=[R["fair"][a]["eod"] for a in attrs]
x=np.arange(len(attrs)); w=0.38
axes[1].bar(x-w/2,cvs,w,label="Cramér's V",color=BLUE)
axes[1].bar(x+w/2,eods,w,label="EOD",color=RED)
axes[1].axhline(0.2,ls="--",color=GREY); axes[1].set_xticks(x); axes[1].set_xticklabels(attrs,rotation=30,fontsize=8)
axes[1].set_title("Cramér's V & EOD"); axes[1].legend()
mi_items=[("StockOptionLevel\n(MaritalStatus)",R["mi"]["MaritalStatus"]["val"]),
          ("TotalWorkingYears\n(Age)",R["mi"]["Age"]["val"]),
          (f'{R["mi"]["Gender"]["top"]}\n(Gender)',R["mi"]["Gender"]["val"])]
axes[2].bar([a for a,_ in mi_items],[b for _,b in mi_items],color=GREEN)
axes[2].set_title("Proxy MI (nats)"); axes[2].tick_params(axis="x",rotation=0,labelsize=7)
plt.tight_layout(); plt.savefig("figures/fig7_fairness.png"); plt.savefig("figures/fig7_fairness.pdf"); plt.savefig("figures/fig7_fairness.svg"); plt.close()

# Fig8 split-robustness: (a) 7-seed check, (b) 100 repeated full-pipeline splits
fig,axes8=plt.subplots(1,2,figsize=(10,4))
axes8[0].plot(range(1,8),seed_aucs,"o-",color=BLUE)
axes8[0].axhline(np.mean(seed_aucs),ls="--",color=GREY,label=f"mean={np.mean(seed_aucs):.3f}")
axes8[0].set_xlabel("Seed index"); axes8[0].set_ylabel("ROC-AUC"); axes8[0].set_ylim(0.7,0.9)
axes8[0].set_title("(a) Seed-sensitivity (7 seeds)"); axes8[0].legend()
axes8[1].hist(rep_split_aucs,bins=15,color=BLUE,alpha=0.75,edgecolor="white")
lo,hi=np.percentile(rep_split_aucs,[2.5,97.5])
axes8[1].axvline(lo,ls="--",color=RED); axes8[1].axvline(hi,ls="--",color=RED,
    label=f"2.5–97.5%: [{lo:.3f}, {hi:.3f}]")
axes8[1].axvline(rep_split_aucs.mean(),ls="-",color=GREY,label=f"mean={rep_split_aucs.mean():.3f}")
axes8[1].set_xlabel("Holdout ROC-AUC"); axes8[1].set_ylabel("Count")
axes8[1].set_title("(b) 100 repeated splits, full refit"); axes8[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig("figures/fig8_seed.png"); plt.savefig("figures/fig8_seed.pdf"); plt.savefig("figures/fig8_seed.svg"); plt.close()

# Fig9 dashboard (6-panel)
fig,axes=plt.subplots(2,3,figsize=(14,8))
# a ROC
from sklearn.metrics import roc_curve
fpr,tpr,_=roc_curve(yte,p_te)
axes[0,0].plot(fpr,tpr,color=BLUE,label=f"AUC={R['holdout_auc']:.3f}")
axes[0,0].plot([0,1],[0,1],"--",color=GREY); axes[0,0].set_title("(a) ROC"); axes[0,0].legend()
# b calibration
axes[0,1].plot([0,1],[0,1],"--",color=GREY); axes[0,1].plot(mean_pred,frac,"o-",color=BLUE)
axes[0,1].set_title(f"(b) Calibration (Brier={R['cv']['brier_mean']:.3f})")
# c ablation
ab=["Full","Comp-\nremoved","Comp-\nonly"]
abv=[R["bench"]["cb_all"]["auc"],R["bench"]["cb_comp_removed"]["auc"],R["bench"]["cb_comp_only"]["auc"]]
axes[0,2].bar(ab,abv,color=[BLUE,BLUE,GREY]); axes[0,2].set_ylim(0.5,0.9); axes[0,2].set_title("(c) Compensation contribution")
for i,v in enumerate(abv): axes[0,2].text(i,v+0.005,f"{v:.3f}",ha="center",fontsize=8)
# d shap
axes[1,0].barh(range(len(tops)),tops.values,color=GREEN); axes[1,0].set_yticks(range(len(tops)))
axes[1,0].set_yticklabels(tops.index,fontsize=6); axes[1,0].set_title("(d) SHAP importance")
# e fairness matrix
axes[1,1].bar(x-w/2,cvs,w,label="V",color=BLUE); axes[1,1].bar(x+w/2,eods,w,label="EOD",color=RED)
axes[1,1].axhline(0.2,ls="--",color=GREY); axes[1,1].set_xticks(x); axes[1,1].set_xticklabels(attrs,rotation=30,fontsize=6)
axes[1,1].set_title("(e) Subgroup fairness"); axes[1,1].legend(fontsize=7)
# f seed
axes[1,2].plot(range(1,8),seed_aucs,"o-",color=BLUE); axes[1,2].set_ylim(0.7,0.9); axes[1,2].set_title("(f) Seed sensitivity")
plt.tight_layout(); plt.savefig("figures/fig9_dashboard.png"); plt.savefig("figures/fig9_dashboard.pdf"); plt.savefig("figures/fig9_dashboard.svg"); plt.close()

json.dump(R, open("results/results.json","w"), indent=2)
print("DONE -> results/results.json")
