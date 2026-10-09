"""McNemar tests on the SANITISED corpus, across every model that has predictions.

Models are aligned on the saved test_idx rather than on position, because the
minimal-text runs drop a few short documents that the stemmed runs keep.
"""
import os, json, glob, itertools
import numpy as np, pandas as pd
from scipy import stats

S = os.path.dirname(os.path.abspath(__file__))
preds = {}  # name -> {test_idx: correct}

z = np.load(os.path.join(S, "out_sanitised_preds.npz"))
y, idx = z["y_true"], z["test_idx"]
for k in z.files:
    if k in ("y_true", "test_idx"):
        continue
    preds[k.replace("_", " ")] = dict(zip(idx, z[k] == y))

for f in sorted(glob.glob(os.path.join(S, "out_indobert_sanitised_seed*.npz"))):
    z = np.load(f)
    seed = os.path.basename(f).split("seed")[1].split(".")[0]
    preds["IndoBERT (seed {})".format(seed)] = dict(zip(z["test_idx"], z["pred"] == z["true"]))

PRETTY = {"indobert_minimal": "IndoBERT (minimal)", "indobertweet": "IndoBERTweet",
          "xlmr": "XLM-R", "mbert": "mBERT"}
seeds = {"IndoBERT (seed 42)": preds.pop("IndoBERT (seed 42)")} if "IndoBERT (seed 42)" in preds else {}
for f in sorted(glob.glob(os.path.join(S, "out_hf_*.npz"))):
    z = np.load(f)
    tag = os.path.basename(f)[len("out_hf_"):-len(".npz")]
    correct = dict(zip(z["test_idx"], z["pred"] == z["true"]))
    if tag.startswith("indobert_seed"):          # replicates, not separate models
        seeds["IndoBERT (seed {})".format(tag[len("indobert_seed"):])] = correct
    else:
        preds[PRETTY.get(tag, tag)] = correct
# the seed-42 run stands in for stemmed IndoBERT in the pairwise comparison
if "IndoBERT (seed 42)" in seeds:
    preds["IndoBERT (stemmed)"] = seeds["IndoBERT (seed 42)"]

print("models with predictions:", list(preds))
print("seed replicates:", list(seeds))


def mcnemar(a, b):
    common = sorted(set(a) & set(b))
    ca = np.array([a[i] for i in common]); cb = np.array([b[i] for i in common])
    n01 = int((ca & ~cb).sum()); n10 = int((~ca & cb).sum()); n = n01 + n10
    if n == 0:
        return len(common), ca.mean(), cb.mean(), n01, n10, np.nan, 1.0
    if n < 25:
        p = float(stats.binomtest(n01, n, 0.5).pvalue); chi2 = np.nan
    else:
        chi2 = (abs(n01 - n10) - 1) ** 2 / n
        p = float(stats.chi2.sf(chi2, 1))
    return len(common), ca.mean(), cb.mean(), n01, n10, chi2, p


rows = []
for a, b in itertools.combinations(preds, 2):
    n, aa, ab, n01, n10, chi2, p = mcnemar(preds[a], preds[b])
    rows.append({"model_a": a, "model_b": b, "n_common": n, "acc_a": aa, "acc_b": ab,
                 "a_right_b_wrong": n01, "a_wrong_b_right": n10, "chi2": chi2, "p_value": p})
df = pd.DataFrame(rows)
order = df["p_value"].rank(method="first").astype(int)
m = len(df)
df["p_holm"] = [min(1.0, (m - order.iloc[i] + 1) * df["p_value"].iloc[i]) for i in range(m)]
df["sig_holm_0.05"] = df["p_holm"] < 0.05
df.to_csv(os.path.join(S, "out_mcnemar_sanitised.csv"), index=False)
print(df.to_string(index=False, float_format=lambda v: "{:.4g}".format(v)))

# seed variance for the headline model (Q12)
if len(seeds) > 1:
    accs = [np.mean(list(v.values())) for v in seeds.values()]
    print("\nIndoBERT seed variance over {} runs: mean {:.4f}  sd {:.4f}  range {:.4f}-{:.4f}".format(
        len(accs), np.mean(accs), np.std(accs, ddof=1), min(accs), max(accs)))
    json.dump({"n_seeds": len(accs), "accuracies": accs,
               "mean": float(np.mean(accs)), "sd": float(np.std(accs, ddof=1))},
              open(os.path.join(S, "out_seed_variance.json"), "w"), indent=2)
print("\nwrote out_mcnemar_sanitised.csv")
