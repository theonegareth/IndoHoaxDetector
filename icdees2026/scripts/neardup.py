"""ICDEES R2 §1: near-duplicate leakage between the training and test splits.

For every test document, find its most similar training document by TF-IDF cosine
similarity (exhaustive, chunked sparse products — no approximation), then measure
the word-trigram Jaccard similarity of that pair, which is the quantity MinHash
estimates. Then re-score every model on the test documents that have NO
near-duplicate in training: if the headline numbers relied on memorised
near-copies, they would fall on this decontaminated subset.
"""
import os, json
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

S = os.path.dirname(os.path.abspath(__file__))
SEED = 42
df = pd.read_csv(os.path.join(S, "deleaked2_preproc.csv"))
X = np.asarray(df["text_clean"].astype(str).tolist(), dtype=object)
y = np.asarray(df["label"].tolist(), dtype=np.int64)
Xtr, Xte, ytr, yte, itr, ite = train_test_split(
    X, y, np.arange(len(y)), test_size=0.2, random_state=SEED, stratify=y)

v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
A, B = v.fit_transform(Xtr), v.transform(Xte)        # rows are L2-normalised
best_sim = np.zeros(B.shape[0]); best_j = np.zeros(B.shape[0], dtype=np.int64)
for s in range(0, B.shape[0], 400):
    sims = (B[s:s + 400] @ A.T).toarray()
    best_j[s:s + 400] = sims.argmax(1); best_sim[s:s + 400] = sims.max(1)
    print("  {}/{}".format(min(s + 400, B.shape[0]), B.shape[0]), end="\r", flush=True)


def shingles(t, k=3):
    w = t.split()
    return {" ".join(w[i:i + k]) for i in range(max(1, len(w) - k + 1))}


jac = np.array([len(a & b) / max(1, len(a | b)) for a, b in
                ((shingles(Xte[i]), shingles(Xtr[best_j[i]])) for i in range(len(Xte)))])
same_label = ytr[best_j] == yte

res = {"n_test": int(len(yte)), "n_train": int(len(ytr)), "similarity": "TF-IDF (1-2 gram) cosine, exhaustive",
       "thresholds": {}}
print("\nnearest-training-neighbour similarity of each test document:")
print("  {:>10} {:>9} {:>8} {:>16}".format("cosine >=", "docs", "%", "label differs"))
for th in (0.95, 0.9, 0.8, 0.7, 0.5):
    m = best_sim >= th
    res["thresholds"][str(th)] = {"n": int(m.sum()), "pct": float(100 * m.mean()),
                                  "n_cross_label": int((m & ~same_label).sum())}
    print("  {:>10} {:>9,} {:>7.2f}% {:>16}".format(th, int(m.sum()), 100 * m.mean(), int((m & ~same_label).sum())))
print("  median nearest cosine {:.3f};  word-trigram Jaccard >= 0.8 for {} test docs ({:.2f}%)".format(
    np.median(best_sim), int((jac >= 0.8).sum()), 100 * (jac >= 0.8).mean()))
res["median_nearest_cosine"] = float(np.median(best_sim))
res["jaccard3_ge_0.8"] = {"n": int((jac >= 0.8).sum()), "pct": float(100 * (jac >= 0.8).mean())}
res["jaccard3_ge_0.5"] = {"n": int((jac >= 0.5).sum()), "pct": float(100 * (jac >= 0.5).mean())}

# re-score on the decontaminated subset (no training neighbour at cosine >= 0.8)
TH = 0.8
keep = best_sim < TH
res["decontaminated"] = {"threshold_cosine": TH, "n_kept": int(keep.sum()), "models": {}}
z = np.load(os.path.join(S, "out_classical_preds_scores.npz"))
assert np.array_equal(z["test_idx"], ite)
preds = {k[:-5].replace("_", " "): (np.ones(len(ite), bool), z[k]) for k in z.files if k.endswith("_pred")}
ib = os.path.join(S, "out_ib_final_seed42.npz")
if os.path.exists(ib):
    # IndoBERT drops docs under 3 words, so its test set is a subset: align by corpus index
    zi = np.load(ib); pos = {j: i for i, j in enumerate(zi["test_idx"])}
    has = np.array([j in pos for j in ite]); p = np.zeros(len(ite), dtype=np.int64)
    p[has] = zi["pred"][[pos[j] for j in ite[has]]]; preds["IndoBERT"] = (has, p)
print("\nre-scored on the {:,} test docs with no training neighbour at cosine >= {}:".format(int(keep.sum()), TH))
for name, (has, p) in preds.items():
    full = f1_score(yte[has], p[has], average="macro")
    sub = f1_score(yte[keep & has], p[keep & has], average="macro")
    res["decontaminated"]["models"][name] = {"macro_f1_full": float(full), "macro_f1_decontaminated": float(sub),
                                             "accuracy_decontaminated": float(accuracy_score(yte[keep & has], p[keep & has]))}
    print("  {:20} macro-F1 {:.4f} -> {:.4f}  ({:+.2f} pts)".format(name, full, sub, 100 * (sub - full)))

np.savez(os.path.join(S, "out_neardup_arrays.npz"), best_sim=best_sim, best_j=best_j, jaccard=jac, test_idx=ite)
with open(os.path.join(S, "out_neardup.json"), "w") as fh:
    json.dump(res, fh, indent=2)
print("wrote out_neardup.json")
