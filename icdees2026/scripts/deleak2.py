"""Aggressive de-leak + the two honest task variants.

Structure of the hoax sources, established by inspection:
  komdigi_hoaks            "Penjelasan : Beredar ... Faktanya ... Kategori : Hoaks ... Link Counter : url"
                           -> the document IS a debunk article; the hoax itself is only quoted inside it.
  turnbackhoax_cleaned     "<claim narrative> Pemeriksaan Fakta ... Kesimpulan ... (Ditulis oleh X)"
                           -> the claim is separable from the fact-check apparatus.
  turnbackhoax_10k_cleaned "<viral post text>" (+ trailing Referensi)
                           -> already the hoax narrative.

Variant A: aggressive de-leak, all sources kept.
Variant B: drop komdigi entirely -> TurnBackHoax claim text vs news text,
           the closest thing in this corpus to genuine hoax-vs-factual.
"""
import os, re, json, time
import pandas as pd, numpy as np
from multiprocessing import Pool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from nltk.corpus import stopwords

S = os.path.dirname(os.path.abspath(__file__))
SEED = 42
STOP = set(stopwords.words("indonesian"))
_cache, _st = {}, None

VERDICT = (r"SALAH|HOAKS?|HOAX|DISINFORMASI|PENIPUAN|MISINFORMASI|FALSE|FITNAH|"
           r"KLARIFIKASI|BENAR|BELUM TERBUKTI|SATIRE|KONTEN TIRUAN|PARODI")
TAG_RE = re.compile(r"^\s*(?:[\[\(]\s*(?:" + VERDICT + r")\s*[\]\)]\s*)+", re.I)
# everything from the fact-check apparatus onward is the checker's voice
CUT_RE = re.compile(r"\b(?:Link\s*Counter|Referensi|Rujukan|Kesimpulan|Pemeriksaan\s*Fakta|"
                    r"Hasil\s*Periksa\s*Fakta|Kategori)\s*:?", re.I)
LEAD_RE = re.compile(r"^\s*Penjelasan\s*:?\s*", re.I)
# debunk-voice sentences
DEBUNK = re.compile(
    r"(?:^|(?<=[.!?]))\s*[^.!?]*\b(?:faktanya|dilansir dari|klaim tersebut|tidak benar|"
    r"hoaks|disinformasi|menyesatkan|tidak ada bukti|beredar sebuah|unggahan tersebut|"
    r"berdasarkan penelusuran|setelah ditelusuri|periksa fakta|mafindo|turnbackhoax|"
    r"cek fakta)\b[^.!?]*[.!?]", re.I)
WRITTEN_BY = re.compile(r"\((?:Ditulis|Penulis)[^)]*\)", re.I)

def deleak(title, text):
    title = TAG_RE.sub("", str(title)).strip()
    t = str(text)
    m = CUT_RE.search(t)
    if m: t = t[:m.start()]
    t = LEAD_RE.sub("", t)
    t = DEBUNK.sub(" ", t)
    t = WRITTEN_BY.sub(" ", t)
    t = re.sub(r"https?://\S+", " ", t)
    return title, re.sub(r"\s+", " ", t).strip()

def _init():
    global _st
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

def evaluate(d, tag, drop_short=3):
    tr, te = train_test_split(d, test_size=0.2, random_state=SEED, stratify=d["label"])
    v = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    A, B = v.fit_transform(tr["text_clean"]), v.transform(te["text_clean"])
    m = LinearSVC(C=1.0, random_state=SEED).fit(A, tr["label"])
    p = m.predict(B); y = te["label"].values
    r = {"n_docs": len(d), "n_test": len(y),
         "accuracy": float(accuracy_score(y, p)),
         "f1_weighted": float(f1_score(y, p, average="weighted", zero_division=0)),
         "confusion_matrix": confusion_matrix(y, p, labels=[0,1]).tolist(),
         "per_class": classification_report(y, p, labels=[0,1], target_names=["factual","hoax"],
                                            output_dict=True, zero_division=0)}
    print(f"\n### {tag}")
    print(f"  n={len(d)}  SVM acc {r['accuracy']:.4f}  F1 {r['f1_weighted']:.4f}")
    print("  confusion [[TN,FP],[FN,TP]] =", r["confusion_matrix"])
    lr = LogisticRegression(C=10, max_iter=1000, random_state=SEED).fit(A, tr["label"])
    names = np.array(v.get_feature_names_out()); co = lr.coef_[0]
    th = [(names[i], round(float(co[i]), 2)) for i in np.argsort(co)[-15:][::-1]]
    tf = [(names[i], round(float(co[i]), 2)) for i in np.argsort(co)[:15]]
    r["top_hoax_features"], r["top_factual_features"] = th, tf
    print("  top hoax   :", ", ".join(w for w, _ in th[:10]))
    print("  top factual:", ", ".join(w for w, _ in tf[:10]))

    # cross-source too
    TR = ["cnn_10k_cleaned","cnn_cleaned","kompas_4k_cleaned","kompas_cleaned",
          "turnbackhoax_cleaned","turnbackhoax_10k_cleaned"]
    TE = ["detik_cleaned","tempo_6k_cleaned","komdigi_hoaks"]
    a, b = d[d.source.isin(TR)], d[d.source.isin(TE)]
    if len(b) and b["label"].nunique() > 1 and a["label"].nunique() > 1:
        v2 = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
        A2 = v2.fit_transform(a["text_clean"])
        m2 = LinearSVC(C=1.0, random_state=SEED).fit(A2, a["label"])
        p2 = m2.predict(v2.transform(b["text_clean"])); y2 = b["label"].values
        r["cross_source"] = {"n": len(y2), "accuracy": float(accuracy_score(y2, p2)),
                             "f1_weighted": float(f1_score(y2, p2, average="weighted", zero_division=0)),
                             "confusion_matrix": confusion_matrix(y2, p2, labels=[0,1]).tolist()}
        print(f"  cross-source: acc {r['cross_source']['accuracy']:.4f}  F1 {r['cross_source']['f1_weighted']:.4f}")
    return r

if __name__ == "__main__":
    t0 = time.time()
    c = pd.read_csv(os.path.join(S, "combined_datasets_clean.csv"))
    tt = [deleak(a, b) for a, b in zip(c["title"], c["text"])]
    c["title_dl"] = [x[0] for x in tt]; c["text_dl"] = [x[1] for x in tt]

    ol = (c["title"].astype(str) + " " + c["text"].astype(str)).str.len()
    nl = (c["title_dl"] + " " + c["text_dl"]).str.len()
    print("=== TEXT REMOVED BY AGGRESSIVE DE-LEAK ===")
    for lab, nm in [(0, "factual"), (1, "hoax")]:
        m = c["label"] == lab
        print(f"  {nm:8} {ol[m].mean():7.0f} -> {nl[m].mean():7.0f} chars ({100*(1-nl[m].mean()/ol[m].mean()):5.1f}% removed)")

    raw = (c["title_dl"] + " " + c["text_dl"]).tolist()
    with Pool(8, initializer=_init) as p:
        c["text_clean"] = p.map(preprocess, raw, chunksize=200)
    c = c[c["text_clean"].str.split().str.len() >= 3].reset_index(drop=True)
    print(f"  kept {len(c)} docs ({time.time()-t0:.0f}s)")
    print("  surviving length:", c.groupby("label")["text_clean"].apply(lambda s: round(s.str.split().str.len().mean(),1)).to_dict())
    c.to_csv(os.path.join(S, "deleaked2_preproc.csv"), index=False)

    res = {}
    res["A_aggressive_all_sources"] = evaluate(c, "VARIANT A: aggressive de-leak, all 9 sources")
    nk = c[c["source"] != "komdigi_hoaks"].reset_index(drop=True)
    res["B_drop_komdigi"] = evaluate(nk, "VARIANT B: drop Komdigi (TurnBackHoax claim text vs news)")

    json.dump(res, open(os.path.join(S, "out_deleaked2.json"), "w"), indent=2)
    print(f"\nwrote out_deleaked2.json ({time.time()-t0:.0f}s)")
