"""ICDEES R2 §6 + §1 (label noise): rule-based error analysis over ALL misclassifications.

Every test error is assigned to the first matching category (priority order below).
Each category's share of errors is compared with its share of the whole test set,
so over-representation (lift) shows which document types the models actually fail on.
Rules are explicit and reproducible; no hand-labelling.
"""
import os, re, json
import numpy as np, pandas as pd

S = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(os.path.join(S, "deleaked2_preproc.csv"))
for c in ("title", "title_dl", "text_dl", "url"):
    d[c] = d[c].fillna("").astype(str)
tag = d["title"].str.extract(r"^\s*[\[\(]\s*([A-Za-z ]{3,25}?)\s*[\]\)]", expand=False).fillna("").str.upper().str.strip()
url = d["url"].str.lower()
minimal = (d["title_dl"] + " " + d["text_dl"]).str.strip()
nwords = minimal.str.split().str.len()
body_extra = d["text_dl"].str.split().str.len() - d["title_dl"].str.split().str.len()
starts_with_title = [t[:40].lower() in b[:60].lower() if t else False for t, b in zip(d["title_dl"], d["text_dl"])]

DEBUNK = re.compile(r"penelusuran|ditelusuri|dibantah|bantahan|membantah|klarifikasi|dilansir|dikutip dari|"
                    r"tidak benar|faktanya|dimanipulasi|disinformasi|hoaks|hoax", re.I)
RUMOUR = re.compile(r"\bisu\b|hoaks|hoax|penipuan|fitnah|bantah|palsu|klarifikasi", re.I)
GENRE = re.compile(r"/kolom|/analisis|/opini|/video|/foto|/infografis")

CATS = [  # (key, description, rule over the dataframe)
    ("label_noise", "Label noise: hoax-labelled doc whose verdict is true/clarification/unproven/news, "
                    "or factual-labelled fact-check article",
     ((d.label == 1) & tag.isin(["BENAR", "KLARIFIKASI", "BELUM TERBUKTI", "BERITA"])) |
     ((d.label == 0) & (url.str.contains("/cekfakta", regex=False) | tag.isin(["HOAKS", "HOAX"])))),
    ("too_short", "Too little content (< 15 words)", nwords < 15),
    ("truncated", "Truncated scrape: body repeats the title plus < 25 words", pd.Series(starts_with_title) & (body_extra < 25)),
    ("non_news_genre", "Opinion, analysis, video or infographic page", url.str.contains(GENRE)),
    ("debunk_residue", "Hoax doc whose surviving text is the fact-checker's rebuttal",
     (d.label == 1) & minimal.str.contains(DEBUNK)),
    ("rumour_topic_news", "Factual news about rumours, fraud or denials",
     (d.label == 0) & minimal.str.contains(RUMOUR)),
]
cat = np.full(len(d), "other", dtype=object)
for key, _, rule in reversed(CATS):       # reversed so the first category wins
    cat[np.asarray(rule, dtype=bool)] = key

models = {}
z = np.load(os.path.join(S, "out_classical_preds_scores.npz"))
models["SVM"] = (z["test_idx"], z["y_true"], z["SVM_pred"])
models["Logistic Regression"] = (z["test_idx"], z["y_true"], z["Logistic_Regression_pred"])
ibf = os.path.join(S, "out_ib_final_seed42.npz")
if os.path.exists(ibf):
    zi = np.load(ibf); models["IndoBERT"] = (zi["test_idx"], zi["true"], zi["pred"])

res = {"categories": {k: desc for k, desc, _ in CATS} | {"other": "None of the above"}, "models": {}}
order = [k for k, _, _ in CATS] + ["other"]
for name, (ti, yt, yp) in models.items():
    err = yt != yp
    base = pd.Series(cat[ti]).value_counts(normalize=True)
    e = pd.Series(cat[ti][err]).value_counts()
    fn = pd.Series(cat[ti][err & (yt == 1)]).value_counts(); fp = pd.Series(cat[ti][err & (yt == 0)]).value_counts()
    rows = {}
    print(f"\n=== {name}: {int(err.sum())} errors of {len(yt)} ({int((err & (yt==1)).sum())} FN, {int((err & (yt==0)).sum())} FP)")
    print(f"  {'category':20} {'errors':>6} {'%err':>6} {'%test':>6} {'lift':>5} {'FN':>4} {'FP':>4}")
    for k in order:
        n = int(e.get(k, 0)); pe = 100 * n / max(1, err.sum()); pt = 100 * base.get(k, 0)
        rows[k] = {"errors": n, "pct_errors": pe, "pct_test": pt, "lift": pe / pt if pt else None,
                   "fn": int(fn.get(k, 0)), "fp": int(fp.get(k, 0))}
        print(f"  {k:20} {n:6d} {pe:5.1f}% {pt:5.1f}% {(pe/pt if pt else 0):5.1f} {rows[k]['fn']:4d} {rows[k]['fp']:4d}")
    res["models"][name] = {"n_errors": int(err.sum()), "n_test": int(len(yt)), "rows": rows}

# a few concrete examples per category for the paper's narrative (from SVM errors)
ti, yt, yp = models["IndoBERT"] if "IndoBERT" in models else models["SVM"]
err_idx = ti[yt != yp]
ex = {}
for k in order:
    pick = [i for i in err_idx if cat[i] == k][:3]
    ex[k] = [{"source": d.source[i], "label": int(d.label[i]), "title": d.title_dl[i][:120]} for i in pick]
res["examples"] = ex
with open(os.path.join(S, "out_error_analysis.json"), "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=2, ensure_ascii=False)
print("\nwrote out_error_analysis.json")
