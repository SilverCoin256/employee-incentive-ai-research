import json, math, statistics
R = json.load(open("results/results.json"))
def g(path):
    d = R
    for k in path.split("/"): d = d[k]
    return d
CHECKS = []
def C(label, ms, path_or_val, dec=3):
    v = g(path_or_val) if isinstance(path_or_val, str) else path_or_val
    CHECKS.append((label, ms, v, dec))
C("n", 1470, "n", 0)
C("base rate %", 16.1, R["base_rate"]*100, 1)
C("CV AUC", 0.815, "cv/auc_mean"); C("CV AUC sd", 0.034, "cv/auc_sd")
C("CV AUC CI lo", 0.786, R["cv"]["auc_ci"][0]); C("CV AUC CI hi", 0.845, R["cv"]["auc_ci"][1])
C("ECE", 0.032, "ece_10bin"); C("ECE CI lo", 0.023, R["ece_ci"][0]); C("ECE CI hi", 0.051, R["ece_ci"][1])
C("LR AUC", 0.827, "bench/lr_unweighted/auc"); C("LR sd", 0.031, "bench/lr_unweighted/auc_sd")
C("LR Brier", 0.095, "bench/lr_unweighted/brier")
C("100split lo", 0.746, R["repeated_splits"]["cb_auc_p2_5_97_5"][0]); C("100split hi", 0.880, R["repeated_splits"]["cb_auc_p2_5_97_5"][1])
C("ablation dAUC", 0.022, R["bench"]["cb_all"]["auc"]-R["bench"]["cb_comp_removed"]["auc"])
C("comp-only AUC", 0.680, "bench/cb_comp_only/auc")
C("comp-removed AUC", 0.793, "bench/cb_comp_removed/auc")
C("EOD gender", 0.109, "fair_full/Gender/eod")
C("EOD gender CI", (0.012,0.237), tuple(g("fair_full/Gender/eod_ci")))
C("EOD ageband", 0.353, "fair_full/Age band/eod")
C("EOD ageband CI", (0.228,0.549), tuple(g("fair_full/Age band/eod_ci")))
C("MI marital", 0.426, "mi/MaritalStatus/val"); C("MI age", 0.582, "mi/Age/val")
C("z marital ~49", 49, round(g("mi_null/MaritalStatus/z")), 0)
C("z age ~49", 49, round(g("mi_null/Age/z")), 0)
C("dummy Brier ref", 0.135, "model_selection/brier_reference_dummy")
C("LR-bal ECE gate fail", 0.200, "bench/lr/ece")
C("selected", "lr_unweighted", "model_selection/selected", None)
C("BSS", 0.245, "threshold_sensitivity/brier/brier_skill_score")
T3 = [
 ("CB 5:1","bench/cb_all",0.815,0.034,0.584,0.102,0.032,0.861),
 ("CB none","bench/cb_all_unweighted",0.814,0.035,0.587,0.102,0.052,0.867),
 ("LR none SEL","bench/lr_unweighted",0.827,0.031,0.614,0.095,0.027,0.876),
 ("LR balanced","bench/lr",0.821,0.036,0.576,0.162,0.200,0.755),
 ("RF balanced","bench/rf",0.810,0.041,0.550,0.106,0.049,0.856),
 ("RF none","bench/rf_unweighted",0.807,0.043,0.562,0.104,0.032,0.859),
 ("CB comp-rem","bench/cb_comp_removed",0.793,0.034,0.555,0.110,0.055,0.854),
 ("CB comp-only","bench/cb_comp_only",0.680,0.015,0.317,0.148,0.094,0.798),
]
for name,base,auc,sd,ap,br,ece,acc in T3:
    C(f"T3 {name} AUC",auc,base+"/auc"); C(f"T3 {name} sd",sd,base+"/auc_sd")
    C(f"T3 {name} AP",ap,base+"/ap"); C(f"T3 {name} Brier",br,base+"/brier")
    C(f"T3 {name} ECE",ece,base+"/ece"); C(f"T3 {name} Acc",acc,base+"/acc")
C("T3 dummy AP",0.161,"bench/dummy/ap"); C("T3 dummy Brier",0.135,"bench/dummy/brier"); C("T3 dummy Acc",0.839,"bench/dummy/acc")
C("T4 Brier CI lo",0.092,R["cv"]["brier_ci"][0]); C("T4 Brier CI hi",0.112,R["cv"]["brier_ci"][1])
apm,aps_=R["cv"]["ap_mean"],R["cv"]["ap_sd"]
C("T4 AP",0.584,apm); C("T4 AP sd",0.053,aps_)
C("T4 AP CI lo",0.538,apm-1.96*aps_/math.sqrt(5)); C("T4 AP CI hi",0.630,apm+1.96*aps_/math.sqrt(5))
am,asd=R["cv"]["acc_mean"],R["cv"]["acc_sd"]
C("T4 Acc",0.861,am); C("T4 Acc sd",0.015,asd)
C("T4 Acc CI lo",0.848,am-1.96*asd/math.sqrt(5)); C("T4 Acc CI hi",0.874,am+1.96*asd/math.sqrt(5))
C("T4 DPD gender",0.011,"fair_full/Gender/dpd")
C("T4 DPD CI",(0.001,0.047),tuple(g("fair_full/Gender/dpd_ci")))
C("CB train",1.00,"overfit_gaps/catboost_weighted/train_auc",2)
C("optimism gap",0.185,"overfit_gaps/catboost_weighted/optimism_gap")
C("LR train",0.870,"overfit_gaps/logreg_unweighted/train_auc")
C("LR CV",0.827,"overfit_gaps/logreg_unweighted/cv_auc")
C("RF CV lo",0.807,"overfit_gaps/rf_unweighted/cv_auc"); C("RF CV hi",0.810,"overfit_gaps/rf_balanced/cv_auc")
sw=R["cb_regularization_sweep"]
C("sweep min train",0.925,min(s["train_auc"] for s in sw))
C("sweep CV lo",0.811,min(s["cv_auc"] for s in sw)); C("sweep CV hi",0.821,max(s["cv_auc"] for s in sw))
C("boot CI lo",0.693,R["bootstrap_auc_ci"][0]); C("boot CI hi",0.859,R["bootstrap_auc_ci"][1])
C("rep CB mean",0.813,"repeated_splits/cb_auc_mean"); C("rep CB sd",0.035,"repeated_splits/cb_auc_sd")
C("rep LR mean",0.833,"repeated_splits/lr_auc_mean")
C("rep LR lo",0.781,R["repeated_splits"]["lr_auc_p2_5_97_5"][0]); C("rep LR hi",0.886,R["repeated_splits"]["lr_auc_p2_5_97_5"][1])
C("rep Brier lo",0.081,R["repeated_splits"]["cb_brier_p2_5_97_5"][0]); C("rep Brier hi",0.116,R["repeated_splits"]["cb_brier_p2_5_97_5"][1])
C("rep ECE lo",0.033,R["repeated_splits"]["cb_ece_p2_5_97_5"][0]); C("rep ECE hi",0.080,R["repeated_splits"]["cb_ece_p2_5_97_5"][1])
C("holdout AUC",0.781,"holdout_auc")
C("ECE adaptive",0.037,"ece_adaptive")
C("mean pred unw",0.114,"mean_pred_unweighted"); C("mean pred w",0.170,"mean_pred_weighted")
C("base rate",0.161,"base_rate")
C("ECE unw",0.052,"ece_10bin_unweighted")
C("ECE unw CI",(0.041,0.071),tuple(g("ece_unweighted_ci")))
C("spearman ref",0.24,"shap_perm_spearman",2)
C("spearman mean",0.25,"xai_spearman/mean",2); C("spearman sd",0.05,"xai_spearman/sd",2)
C("spearman min",0.14,"xai_spearman/min",2); C("spearman max",0.33,"xai_spearman/max",2)
C("mask top5 mean",0.070,"xai_masking/top5_dauc_mean"); C("mask top5 sd",0.031,"xai_masking/top5_dauc_sd")
C("mask rand mean",0.018,"xai_masking/random5_dauc_mean"); C("mask rand sd",0.020,"xai_masking/random5_dauc_sd")
C("mask z",2.6,"xai_masking/z_vs_random",1)
C("shuffled AUC",0.486,"xai_shuffled_label/auc_mean"); C("shuffled sd",0.046,"xai_shuffled_label/auc_sd")
C("shuffled maxdrop",0.025,"xai_shuffled_label/maxdrop_mean"); C("shuffled maxdrop sd",0.009,"xai_shuffled_label/maxdrop_sd")
C("perm top1 (ms claims 0.034-0.070)",0.034,R["perm_top"][0][1])
C("MI gender",0.031,"mi/Gender/val")
C("MI gender re-est",0.015,"mi_null/Gender/observed")
C("z gender",1.4,"mi_null/Gender/z",1); C("p gender",0.10,"mi_null/Gender/p_perm",2)
C("MI marital re-est",0.419,"mi_null/MaritalStatus/observed")
C("null mean marital",0.006,"mi_null/MaritalStatus/null_mean")
C("p marital",0.005,"mi_null/MaritalStatus/p_perm")
C("MI age re-est",0.585,"mi_null/Age/observed")
T5 = [
 ("Gender",0.908,(0.667,0.995),0.015,(0.000,0.070),0.011,(0.001,0.047),0.109,(0.012,0.237),(588,882),(87,150)),
 ("Age band",0.213,(0.063,0.354),0.174,(0.125,0.226),0.155,(0.110,0.204),0.353,(0.228,0.549),(143,619),(18,100)),
 ("Compensation band",0.117,(0.051,0.189),0.281,(0.231,0.336),0.225,(0.180,0.275),0.399,(0.259,0.551),(366,369),(38,108)),
 ("Marital status",0.284,(0.163,0.416),0.183,(0.132,0.236),0.139,(0.098,0.179),0.190,(0.090,0.336),(327,673),(33,120)),
 ("Department",0.542,(0.328,0.800),0.075,(0.028,0.136),0.080,(0.026,0.185),0.073,(0.050,0.410),(63,961),(12,133)),
]
for attr,dir_,dirci,v,vci,dpd,dpdci,eod,eodci,nrange,posrange in T5:
    b=f"fair_full/{attr}"
    C(f"T5 {attr} DIR",dir_,b+"/dir"); C(f"T5 {attr} DIR CI",dirci,tuple(g(b+"/dir_ci")))
    C(f"T5 {attr} V",v,b+"/cv");      C(f"T5 {attr} V CI",vci,tuple(g(b+"/cv_ci")))
    C(f"T5 {attr} DPD",dpd,b+"/dpd"); C(f"T5 {attr} DPD CI",dpdci,tuple(g(b+"/dpd_ci")))
    C(f"T5 {attr} EOD",eod,b+"/eod"); C(f"T5 {attr} EOD CI",eodci,tuple(g(b+"/eod_ci")))
    grp=g(b+"/groups"); ns=sorted(x["n"] for x in grp.values()); ps=sorted(x["n_pos"] for x in grp.values())
    C(f"T5 {attr} n range",nrange,(ns[0],ns[-1]),0); C(f"T5 {attr} pos range",posrange,(ps[0],ps[-1]),0)
C("LR V comp",0.177,"fair_lr_unweighted/Compensation band/cv")
C("LR EOD marital",0.221,"fair_lr_unweighted/Marital status/eod")
ts=R["threshold_sensitivity"]
C("V flags @0.20",["Compensation band"],ts["cramers_v"]["grid"]["0.20"],None)
C("V flags @0.30",[],ts["cramers_v"]["grid"]["0.30"],None)
C("EOD @0.20",["Age band","Compensation band"],sorted(ts["eod"]["grid"]["0.20"]),None)
C("EOD @0.15",["Age band","Compensation band","Marital status"],sorted(ts["eod"]["grid"]["0.15"]),None)
C("EOD @0.10",["Age band","Compensation band","Gender","Marital status"],sorted(ts["eod"]["grid"]["0.10"]),None)
C("DIR stable 0.60-0.90",True,all(sorted(ts["dir"]["grid"][k])==["Age band","Compensation band","Department","Marital status"] for k in ["0.60","0.70","0.80","0.90"]),None)
C("ECE passes to 0.04",True,ts["ece"]["grid"]["0.04"] and not ts["ece"]["grid"]["0.03"],None)
C("sil KMeans",0.116,"seg/KMeans"); C("sil PCA",0.164,"seg/PCA10+KMeans"); C("sil GMM",0.087,"seg/GMM")
C("PCA var %",69.0,R["pca_var10"]*100,1)
C("seed min",0.781,"seed_min"); C("seed max",0.845,"seed_max")
C("seed mean",0.811,statistics.mean(R["seed_aucs"]))
fails=[]
for label,ms,v,dec in CHECKS:
    if dec is None:
        ok = ms==v; shown=v
    elif isinstance(ms,tuple):
        shown=tuple(round(x,3) for x in v)
        ok = all(abs(round(x,3)-m)<5e-4+1e-9 for x,m in zip(v,ms))
    else:
        shown=round(v,dec) if dec else round(v)
        ok = abs(round(v,dec)-ms) < 10**(-dec)/2+1e-9 if dec else round(v)==ms
    if not ok: fails.append((label,ms,shown))
print(f"{len(CHECKS)} checks, {len(fails)} failures")
for f in fails: print("  FAIL:",f)
