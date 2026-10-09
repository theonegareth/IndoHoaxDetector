"""External validity experiments (reviewer R1#3, R2 Q6/Q7/Q8/Q9/Q16).

  A. random split           -- in-distribution reference
  B. grouped cross-source   -- disjoint outlets, both classes in train and test
  C. leave-one-source-out   -- per-source recall when that outlet is unseen
  D. X.com transfer         -- 10k tweets from two mainstream news accounts;
                               these should read as factual, so the hoax rate
                               is an out-of-domain false-positive rate.
"""
import os, re, json, time
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix, classification_report
from nltk.corpus import stopwords

S = os.path.dirname(os.path.abspath(__file__))
SEED = 42
STOP = set(stopwords.words("indonesian"))
_cache = {}
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
_st = StemmerFactory().create_stemmer()

def preprocess(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", str(text).lower(), flags=re.MULTILINE)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    out = []
    for w in text.split():
        if w in STOP: continue
        s = _cache.get(w)
        if s is None: s = _cache[w] = _st.stem(w)
        out.append(s)
    return " ".join(out)

# CORPUS=sanitised runs everything on the de-leaked corpus the paper reports;
# the default is the original corpus, kept for the appendix comparison.
SANITISED = os.environ.get("CORPUS") == "sanitised"
if SANITISED:
    df = pd.read_csv(os.path.join(S, "deleaked2_preproc.csv")).rename(columns={"label": "label_encoded"})
else:
    df = pd.read_csv(os.path.join(S, "preproc_with_meta.csv"))
OUTFILE = "out_cross_source_sanitised.json" if SANITISED else "out_cross_source.json"
df["text_clean"] = df["text_clean"].astype(str)
MF, NG = 10000, (1, 2)

def fit(tr):
    v = TfidfVectorizer(max_features=MF, ngram_range=NG)
    A = v.fit_transform(tr["text_clean"])
    m = LinearSVC(C=1.0, random_state=SEED).fit(A, tr["label_encoded"])
    return v, m

def ev(v, m, te, name, verbose=True):
    p = m.predict(v.transform(te["text_clean"]))
    y = te["label_encoded"].values
    r = {"n": len(y), "accuracy": float(accuracy_score(y, p)),
         "f1_weighted": float(f1_score(y, p, average="weighted", zero_division=0)),
         "confusion_matrix": confusion_matrix(y, p, labels=[0,1]).tolist(),
         "per_class": classification_report(y, p, labels=[0,1], target_names=["factual","hoax"],
                                            output_dict=True, zero_division=0)}
    if verbose:
        print(f"\n### {name}\n  n={len(y)}  acc={r['accuracy']:.4f}  F1={r['f1_weighted']:.4f}")
        print("  confusion [[TN,FP],[FN,TP]] =", r["confusion_matrix"])
        pc = r["per_class"]
        print(f"  factual: P={pc['factual']['precision']:.4f} R={pc['factual']['recall']:.4f} | "
              f"hoax: P={pc['hoax']['precision']:.4f} R={pc['hoax']['recall']:.4f}")
    return r, p

res = {}

# ---- A. random split ------------------------------------------------------
tr, te = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["label_encoded"])
v, m = fit(tr); res["A_random"], _ = ev(v, m, te, "A. Random 80/20 split (in-distribution)")

# ---- B. grouped cross-source ---------------------------------------------
TR = ["cnn_10k_cleaned","cnn_cleaned","kompas_4k_cleaned","kompas_cleaned",
      "turnbackhoax_cleaned","turnbackhoax_10k_cleaned"]
TE = ["detik_cleaned","tempo_6k_cleaned","komdigi_hoaks"]
trb, teb = df[df["source"].isin(TR)], df[df["source"].isin(TE)]
vb, mb = fit(trb)
res["B_cross_source"], _ = ev(vb, mb, teb, "B. Cross-source (train CNN+Kompas+TBH -> test Detik+Tempo+Komdigi)")
res["B_cross_source"]["train_sources"], res["B_cross_source"]["test_sources"] = TR, TE
per = {}
for s in TE:
    sub = teb[teb["source"] == s]
    p = mb.predict(vb.transform(sub["text_clean"]))
    per[s] = {"n": len(sub), "label": int(sub["label_encoded"].iloc[0]),
              "accuracy": float(accuracy_score(sub["label_encoded"], p))}
res["B_cross_source"]["per_source"] = per
print("  per test source:", json.dumps(per))

# ---- C. leave-one-source-out ---------------------------------------------
print("\n### C. Leave-one-source-out (each outlet held out entirely)")
print(f"{'held-out source':26} {'label':8} {'n':>6} {'recall unseen':>14} {'recall in-dist':>15}")
loso = {}
for s in sorted(df["source"].unique()):
    tr_s, te_s = df[df["source"] != s], df[df["source"] == s]
    lab = int(te_s["label_encoded"].iloc[0])
    vs, ms = fit(tr_s)
    rec_unseen = float(accuracy_score(te_s["label_encoded"], ms.predict(vs.transform(te_s["text_clean"]))))
    # in-distribution reference: same outlet, but model trained on everything
    idx = te_s.sample(min(len(te_s), 2000), random_state=SEED)
    rec_in = float(accuracy_score(idx["label_encoded"], m.predict(v.transform(idx["text_clean"]))))
    loso[s] = {"label": lab, "n": len(te_s), "recall_when_unseen": rec_unseen,
               "recall_in_distribution": rec_in}
    print(f"{s:26} {'hoax' if lab else 'factual':8} {len(te_s):6d} {rec_unseen:14.4f} {rec_in:15.4f}")
res["C_leave_one_source_out"] = loso

# ---- D. X.com transfer ----------------------------------------------------
print("\n### D. X.com transfer test (10k tweets from @kompascom and @VIVAcoid)")
tw = []
for f, acct in [("tweets_from_kompascom_20251106_124453.csv", "kompascom"),
                ("tweets_from_VIVAcoid_20251106_122602.csv", "VIVAcoid")]:
    t = pd.read_csv(os.path.join(S, f)); t["account"] = acct; tw.append(t)
tw = pd.concat(tw, ignore_index=True)
tw = tw[tw["lang"].astype(str).str.startswith("in")].reset_index(drop=True)
tw["text_clean"] = tw["text"].map(preprocess)
tw = tw[tw["text_clean"].str.split().str.len() >= 3].reset_index(drop=True)
print(f"  {len(tw)} Indonesian-language tweets from mainstream news accounts")
print(f"  mean length {tw['text_clean'].str.split().str.len().mean():.1f} tokens "
      f"(corpus factual mean = {df[df.label_encoded==0]['text_clean'].str.split().str.len().mean():.1f})")

tweet_res = {}
p_all = m.predict(v.transform(tw["text_clean"]))          # model trained on the random split
tweet_res["overall_hoax_rate"] = float(p_all.mean())
print(f"\n  SVM trained on full corpus flags {p_all.mean()*100:.1f}% of these tweets as HOAX")
for acct in tw["account"].unique():
    mask = (tw["account"] == acct).values
    tweet_res[acct] = {"n": int(mask.sum()), "hoax_rate": float(p_all[mask].mean())}
    print(f"    @{acct:12} n={mask.sum():5d}  hoax-flag rate {p_all[mask].mean()*100:5.1f}%")

# length-matched control: corpus factual docs truncated to tweet length
tl = int(tw["text_clean"].str.split().str.len().median())
fac = df[df["label_encoded"] == 0].sample(4000, random_state=SEED).copy()
fac["trunc"] = fac["text_clean"].str.split().str[:tl].str.join(" ")
p_tr = m.predict(v.transform(fac["trunc"]))
tweet_res["length_matched_control_hoax_rate"] = float(p_tr.mean())
print(f"\n  CONTROL: real factual news truncated to {tl} tokens -> {p_tr.mean()*100:.1f}% flagged hoax")
print("  (isolates how much of the tweet result is length/truncation rather than platform)")
res["D_xcom_transfer"] = tweet_res

json.dump(res, open(os.path.join(S, OUTFILE), "w"), indent=2)
print("corpus:", "sanitised" if SANITISED else "original")
print("\nwrote out_cross_source.json")
