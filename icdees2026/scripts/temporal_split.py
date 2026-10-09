"""Temporal validation (closes R1 #3 'temporal validation' and R2 Q6 'time-separated')
and per-source performance (R2 Q7).

Publication dates are recoverable from the URL for six of the nine source files,
covering both classes, so a genuine train-on-past / test-on-future split is
possible over that subset even though the whole corpus cannot be dated.
"""
import os, re, json
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

S = os.path.dirname(os.path.abspath(__file__))
SEED = 42
df = pd.read_csv(os.path.join(S, "deleaked2_preproc.csv"))
df["text_clean"] = df["text_clean"].astype(str)
df["url"] = df["url"].astype(str)

# ---- recover publication dates from URLs ----------------------------------
# Slashed dates first; the compact form only in CNN's exact 14-digit shape. A looser
# compact pattern misreads Kompas article IDs, which are publish times (/20104261 = 20:10:42).
PATS = [r"/(20\d{2})/(\d{2})/(\d{2})/", r"/(20\d{2})/(\d{2})/", r"/(20\d{2})(\d{2})(\d{2})\d{6}-"]


def ym(u):
    for p in PATS:
        m = re.search(p, u)
        if m:
            g = m.groups()
            y, mo = int(g[0]), int(g[1]) if len(g) > 1 else 1
            if 2010 <= y <= 2026 and 1 <= mo <= 12:
                return y * 100 + mo
    return np.nan


df["ym"] = df["url"].map(ym)
dated = df[df["ym"].notna()].copy()
print("dated documents: {:,} of {:,} ({:.1%})".format(len(dated), len(df), len(dated) / len(df)))
print(dated.groupby("label")["ym"].agg(["count", "min", "max"]))

res = {}

# ---- temporal split -------------------------------------------------------
CUT = 202301
tr, te = dated[dated.ym < CUT], dated[dated.ym >= CUT]
print("\ncutoff {}: train {:,} (to {}), test {:,} (from {})".format(
    CUT, len(tr), int(tr.ym.max()), len(te), int(te.ym.min())))
print("  train labels", tr.label.value_counts().to_dict(),
      " test labels", te.label.value_counts().to_dict())

v = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
A = v.fit_transform(tr["text_clean"])
m = LinearSVC(C=1.0, random_state=SEED).fit(A, tr["label"])
p = m.predict(v.transform(te["text_clean"]))
y = te["label"].values
res["temporal"] = {
    "cutoff": CUT, "n_train": len(tr), "n_test": len(te),
    "accuracy": float(accuracy_score(y, p)),
    "f1_weighted": float(f1_score(y, p, average="weighted", zero_division=0)),
    "confusion_matrix": confusion_matrix(y, p, labels=[0, 1]).tolist(),
    "per_class": classification_report(y, p, labels=[0, 1], target_names=["factual", "hoax"],
                                       output_dict=True, zero_division=0),
}
print("  TEMPORAL  acc {:.4f}  F1 {:.4f}".format(res["temporal"]["accuracy"],
                                                 res["temporal"]["f1_weighted"]))

# random-split control on the same dated subset, so the comparison is like-for-like
rtr, rte = train_test_split(dated, test_size=len(te) / len(dated), random_state=SEED,
                            stratify=dated["label"])
v2 = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
A2 = v2.fit_transform(rtr["text_clean"])
m2 = LinearSVC(C=1.0, random_state=SEED).fit(A2, rtr["label"])
p2 = m2.predict(v2.transform(rte["text_clean"]))
res["random_control_same_subset"] = {
    "n_train": len(rtr), "n_test": len(rte),
    "accuracy": float(accuracy_score(rte["label"], p2)),
    "f1_weighted": float(f1_score(rte["label"], p2, average="weighted", zero_division=0)),
}
print("  RANDOM control on the same documents  acc {:.4f}  F1 {:.4f}".format(
    res["random_control_same_subset"]["accuracy"],
    res["random_control_same_subset"]["f1_weighted"]))

# ---- Q7: per-source performance of the delivered models -------------------
npz = np.load(os.path.join(S, "out_sanitised_preds.npz"))
y_all, idx = npz["y_true"], npz["test_idx"]
src = df["source"].values[idx]
per = {}
for k in npz.files:
    if k in ("y_true", "test_idx"):
        continue
    name = k.replace("_", " ")
    pr = npz[k]
    per[name] = {s: {"n": int((src == s).sum()),
                     "accuracy": float(accuracy_score(y_all[src == s], pr[src == s]))}
                 for s in np.unique(src)}
# headline IndoBERT = minimal preprocessing; its test_idx indexes the same dataframe
ibp = os.path.join(S, "out_hf_indobert_minimal.npz")
if not os.path.exists(ibp):
    ibp = os.path.join(S, "out_indobert_sanitised_seed42.npz")
if os.path.exists(ibp):
    z = np.load(ibp)
    s2 = df["source"].values[z["test_idx"]]
    per["IndoBERT"] = {s: {"n": int((s2 == s).sum()),
                           "accuracy": float(accuracy_score(z["true"][s2 == s], z["pred"][s2 == s]))}
                       for s in np.unique(s2)}
res["per_source"] = per

print("\nper-source accuracy on the sanitised held-out set (Q7):")
srcs = sorted({s for d in per.values() for s in d})
print("{:26} {:>7}".format("source", "n"), "".join("{:>14}".format(m[:13]) for m in per))
for s in srcs:
    n = next(d[s]["n"] for d in per.values() if s in d)
    print("{:26} {:>7,}".format(s, n),
          "".join("{:>14.4f}".format(per[m][s]["accuracy"]) if s in per[m] else " " * 14
                  for m in per))

with open(os.path.join(S, "out_temporal_and_per_source.json"), "w") as fh:
    json.dump(res, fh, indent=2)
print("\nwrote out_temporal_and_per_source.json")
