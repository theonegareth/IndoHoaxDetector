"""Summarise the RunPod IndoBERT runs (pod/pod_results/*/out_pod_*.json|npz) with the local runs.

Writes out_extra_ib.json:
  xsource      IndoBERT publisher-disjoint, seeds 42 (local) + 1, 2 (pod): weighted F1 mean/sd, per-source accuracy mean/sd
  loso         IndoBERT recall per withheld source vs SVM (same splits) and vs IndoBERT seeing the source (random split)
  topic, time  IndoBERT drop against the matched random control (same documents and size), next to SVM's
  encoders     accuracy, macro-F1 and McNemar (Holm over the three) against IndoBERT seed 42 on the shared test set
  lr_sweep     validation and test weighted F1 per learning rate; the rate validation selects
"""
import os, glob, json, math
import numpy as np, pandas as pd

S = os.path.dirname(os.path.abspath(__file__))
R = {os.path.basename(f)[8:-5]: json.load(open(f)) for f in glob.glob(os.path.join(S, "pod", "pod_results", "*", "out_pod_*.json"))}
NPZ = {os.path.basename(f)[8:-4]: np.load(f) for f in glob.glob(os.path.join(S, "pod", "pod_results", "*", "out_pod_*.npz"))}
src = pd.read_csv(os.path.join(S, "deleaked2_preproc.csv"), usecols=["source"])["source"].values
J = lambda f: json.load(open(os.path.join(S, f)))
out, missing = {}, []
need = lambda *tags: [t for t in tags if t not in R]

# --- publisher-disjoint, three seeds -------------------------------------------------------------------------
xs42 = J("out_ib_final_seed42_xsource.json"); z42 = np.load(os.path.join(S, "out_ib_final_seed42_xsource.npz"))
runs = [(xs42["f1_weighted"], z42["pred"], z42["true"], z42["test_idx"])]
for t in ("xs_s1", "xs_s2"):
    if t in R: runs.append((R[t]["f1_weighted"], NPZ[t]["pred"], NPZ[t]["true"], NPZ[t]["test_idx"]))
missing += need("xs_s1", "xs_s2")
per = {}
for s in ("detik_cleaned", "tempo_6k_cleaned", "komdigi_hoaks"):
    accs = [float((p[src[ix] == s] == y[src[ix] == s]).mean()) for _, p, y, ix in runs]
    per[s] = {"mean": float(np.mean(accs)), "sd": float(np.std(accs, ddof=1)) if len(accs) > 1 else None, "runs": accs}
f1s = [r[0] for r in runs]
out["xsource"] = {"n_seeds": len(runs), "f1_weighted_mean": float(np.mean(f1s)), "f1_weighted_sd": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else None,
                  "f1_weighted_runs": f1s, "per_source": per}

# --- leave-one-source-out ------------------------------------------------------------------------------------
svm_loso = J("out_cross_source_sanitised.json")["C_leave_one_source_out"]
zr = np.load(os.path.join(S, "out_ib_final_seed42.npz"))
SEEDS = ("", "_s1", "_s2")                     # seed 42 (first batch), seeds 1 and 2
sd = lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else None
loso = {}
for s, v in svm_loso.items():
    tags = [t for t in ("loso_" + s + x for x in SEEDS) if t in R]
    missing += [t for t in ("loso_" + s + x for x in SEEDS) if t not in R]
    if not tags: continue
    rec = [R[t]["accuracy"] for t in tags]
    seen = src[zr["test_idx"]] == s
    loso[s] = {"n": R[tags[0]]["n_test"], "n_seeds": len(rec), "ib_recall_unseen": float(np.mean(rec)), "ib_recall_unseen_sd": sd(rec),
               "ib_recall_unseen_runs": rec, "ib_recall_seen": float((zr["pred"][seen] == zr["true"][seen]).mean()),
               "svm_recall_unseen": v["recall_when_unseen"], "svm_recall_seen": v["recall_in_distribution"]}
if loso:
    drop = lambda k: [100 * (d[k.replace("unseen", "seen")] - d[k]) for d in loso.values()]
    out["loso"] = {"per_source": loso, "ib_drop_min": min(drop("ib_recall_unseen")), "ib_drop_max": max(drop("ib_recall_unseen")),
                   "svm_drop_min": min(drop("svm_recall_unseen")), "svm_drop_max": max(drop("svm_recall_unseen")),
                   "ib_better_on": [s for s, d in loso.items() if d["ib_recall_unseen"] > d["svm_recall_unseen"] + 0.0005],
                   "ib_worse_on": [s for s, d in loso.items() if d["ib_recall_unseen"] < d["svm_recall_unseen"] - 0.0005]}

# --- topic and time: drop against the control trained with the same seed ---------------------------------------
svm_ref = {"topic": (0.9654, 0.9756), "time": (0.9632, 0.9930)}       # SVM weighted F1, held out vs control (Table V)
for k in ("topic", "time"):
    pairs = [(R[k + x], R[k + "_ctl" + x]) for x in SEEDS if k + x in R and k + "_ctl" + x in R]
    missing += [t for x in SEEDS for t in (k + x, k + "_ctl" + x) if t not in R]
    if not pairs: continue
    drops = [100 * (c["f1_weighted"] - h["f1_weighted"]) for h, c in pairs]
    out[k] = {"n_seeds": len(pairs), "ib_f1": float(np.mean([h["f1_weighted"] for h, _ in pairs])),
              "ib_ctl": float(np.mean([c["f1_weighted"] for _, c in pairs])), "ib_drop": float(np.mean(drops)), "ib_drop_sd": sd(drops),
              "ib_drop_runs": drops, "svm_f1": svm_ref[k][0], "svm_ctl": svm_ref[k][1], "svm_drop": 100 * (svm_ref[k][1] - svm_ref[k][0])}

# --- encoders vs IndoBERT seed 42 (McNemar, continuity-corrected, Holm over the three) -------------------------
pos = {j: i for i, j in enumerate(zr["test_idx"])}
enc, ps = {}, {}
for t in ("enc_indobertweet", "enc_mbert", "enc_xlmr"):
    if t not in R: missing.append(t); continue
    z = NPZ[t]; ii = np.array([pos[j] for j in z["test_idx"]]); y = z["true"]
    a, b = (zr["pred"][ii] == y), (z["pred"] == y)
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    x = (abs(n01 - n10) - 1) ** 2 / (n01 + n10) if n01 + n10 else 0.0
    ps[t] = math.erfc(math.sqrt(x / 2))
    enc[t] = {"model": R[t]["model"], "accuracy": R[t]["accuracy"], "f1_macro": R[t]["f1_macro"], "roc_auc": R[t]["roc_auc"],
              "ib_right_other_wrong": n01, "ib_wrong_other_right": n10}
keys = sorted(ps, key=ps.get); run = 0.0
for i, k in enumerate(keys):
    run = max(run, min(1.0, (len(keys) - i) * ps[k])); enc[k]["p_holm_vs_indobert"] = run
if enc: out["encoders"] = enc

# --- learning-rate sweep -------------------------------------------------------------------------------------
lr = {t: {"lr": R[t]["lr"], "val_f1": R[t]["val_f1_weighted_best"], "test_f1": R[t]["f1_weighted"], "test_acc": R[t]["accuracy"]}
      for t in ("lr1e5", "lr2e5", "lr3e5", "lr5e5") if t in R}
missing += need("lr1e5", "lr2e5", "lr3e5", "lr5e5")
if lr:
    best = max(lr.values(), key=lambda v: v["val_f1"])
    out["lr_sweep"] = {"runs": lr, "selected_lr": best["lr"], "selected_test_acc": best["test_acc"],
                       "val_f1_spread": 100 * (max(v["val_f1"] for v in lr.values()) - min(v["val_f1"] for v in lr.values()))}

out["missing"] = sorted(set(missing))
json.dump(out, open(os.path.join(S, "out_extra_ib.json"), "w"), indent=2)
print(json.dumps(out, indent=1)[:4000])
