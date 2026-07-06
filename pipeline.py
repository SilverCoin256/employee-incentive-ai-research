"""
Single reproducible pipeline for the HR-attrition governance-audit manuscript.
Produces EVERY reported number and figure from one script/one dataset/one model version.
Run: python3 pipeline.py  -> writes results.json and figs/*.png
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
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr
from catboost import CatBoostClassifier, Pool
import os
os.makedirs("figs", exist_ok=True)
SEED = 42
rng = np.random.RandomState(SEED)
R = {}  # results dict

# ---------- load ----------
df = pd.read_csv("hr.csv")
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
def eval_cb(cols):
    oof_,a,b,ap,ac,_=catboost_cv_oof(X_all,y,cols)
    return dict(auc=float(a.mean()),ap=float(ap.mean()),brier=float(b.mean()),acc=float(ac.mean()))
comp_removed=[c for c in X_all.columns if c not in COMP]
R["bench"]={}
R["bench"]["cb_all"]=dict(auc=R["cv"]["auc_mean"],ap=R["cv"]["ap_mean"],brier=R["cv"]["brier_mean"],acc=R["cv"]["acc_mean"])
R["bench"]["cb_comp_removed"]=eval_cb(comp_removed)
R["bench"]["cb_comp_only"]=eval_cb(COMP)

# LR & RF need numeric encoding
pre=ColumnTransformer([("num",StandardScaler(),num_cols),
                       ("cat",OneHotEncoder(handle_unknown="ignore"),cat_cols)])
def eval_sklearn(est):
    skf=StratifiedKFold(5,shuffle=True,random_state=SEED)
    a=[];ap=[];br=[];ac=[]
    for tr,va in skf.split(X_all,y):
        pipe=Pipeline([("pre",pre),("clf",est)])
        pipe.fit(X_all.iloc[tr],y[tr])
        p=pipe.predict_proba(X_all.iloc[va])[:,1]
        a.append(roc_auc_score(y[va],p));ap.append(average_precision_score(y[va],p))
        br.append(brier_score_loss(y[va],p));ac.append(accuracy_score(y[va],(p>=.5).astype(int)))
    return dict(auc=float(np.mean(a)),ap=float(np.mean(ap)),brier=float(np.mean(br)),acc=float(np.mean(ac)))
R["bench"]["lr"]=eval_sklearn(LogisticRegression(max_iter=1000,C=1.0,class_weight="balanced"))
R["bench"]["rf"]=eval_sklearn(RandomForestClassifier(n_estimators=400,min_samples_leaf=2,random_state=SEED,class_weight="balanced"))
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
perm=permutation_importance(mh, Xte, yte, n_repeats=10, random_state=SEED,
                            scoring="roc_auc")
# Xte has categoricals as object -> catboost predict handles; permutation_importance needs numeric predict? use wrapper
# fallback: compute perm on catboost via manual
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
plt.tight_layout(); plt.savefig("figs/fig2_segmentation.png"); plt.savefig("figs/fig2_segmentation.pdf"); plt.savefig("figs/fig2_segmentation.svg"); plt.close()

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
plt.tight_layout(); plt.savefig("figs/fig3_confusion.png"); plt.savefig("figs/fig3_confusion.pdf"); plt.savefig("figs/fig3_confusion.svg"); plt.close()
R["cm"]=cm.tolist()

# Fig4 calibration curve
from sklearn.calibration import calibration_curve
frac,mean_pred=calibration_curve(y,oof,n_bins=10,strategy="uniform")
fig,ax=plt.subplots(figsize=(5,4))
ax.plot([0,1],[0,1],"--",color=GREY,label="perfect")
ax.plot(mean_pred,frac,"o-",color=BLUE,label=f"CatBoost (ECE={R['ece_10bin']:.3f})")
ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed frequency")
ax.set_title(f"Calibration curve (Brier={R['cv']['brier_mean']:.3f})"); ax.legend()
plt.tight_layout(); plt.savefig("figs/fig4_calibration.png"); plt.savefig("figs/fig4_calibration.pdf"); plt.savefig("figs/fig4_calibration.svg"); plt.close()

# Fig5 permutation importance
fig,ax=plt.subplots(figsize=(6,4))
top=perm_s.head(10)[::-1]
ax.barh(range(len(top)),top.values,color=BLUE)
ax.set_yticks(range(len(top))); ax.set_yticklabels(top.index,fontsize=8)
ax.set_xlabel("Mean AUC drop"); ax.set_title("Permutation importance (holdout)")
plt.tight_layout(); plt.savefig("figs/fig5_permutation.png"); plt.savefig("figs/fig5_permutation.pdf"); plt.savefig("figs/fig5_permutation.svg"); plt.close()

# Fig6 SHAP summary (bar of mean|SHAP|)
fig,ax=plt.subplots(figsize=(6,4))
tops=shap_s.head(10)[::-1]
ax.barh(range(len(tops)),tops.values,color=GREEN)
ax.set_yticks(range(len(tops))); ax.set_yticklabels(tops.index,fontsize=8)
ax.set_xlabel("mean |SHAP value|"); ax.set_title("SHAP feature importance (holdout)")
plt.tight_layout(); plt.savefig("figs/fig6_shap.png"); plt.savefig("figs/fig6_shap.pdf"); plt.savefig("figs/fig6_shap.svg"); plt.close()

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
plt.tight_layout(); plt.savefig("figs/fig7_fairness.png"); plt.savefig("figs/fig7_fairness.pdf"); plt.savefig("figs/fig7_fairness.svg"); plt.close()

# Fig8 seed sensitivity
fig,ax=plt.subplots(figsize=(6,4))
ax.plot(range(1,8),seed_aucs,"o-",color=BLUE)
ax.axhline(np.mean(seed_aucs),ls="--",color=GREY,label=f"mean={np.mean(seed_aucs):.3f}")
ax.set_xlabel("Seed index"); ax.set_ylabel("ROC-AUC"); ax.set_ylim(0.7,0.9)
ax.set_title("Seed-sensitivity (7 seeds)"); ax.legend()
plt.tight_layout(); plt.savefig("figs/fig8_seed.png"); plt.savefig("figs/fig8_seed.pdf"); plt.savefig("figs/fig8_seed.svg"); plt.close()

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
plt.tight_layout(); plt.savefig("figs/fig9_dashboard.png"); plt.savefig("figs/fig9_dashboard.pdf"); plt.savefig("figs/fig9_dashboard.svg"); plt.close()

json.dump(R, open("results.json","w"), indent=2)
print("DONE")
print(json.dumps(R,indent=1)[:60])
