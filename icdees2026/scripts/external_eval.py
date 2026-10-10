"""External benchmark: our models, unchanged, on the Pratiwi et al. (2017) Indonesian Hoax News Detection
Dataset (Mendeley Data, doi 10.17632/p3hfgr5j3m.1, CC BY 4.0): 600 online news articles, each labelled
hoax or valid by majority vote of three human annotators.

Nothing is trained on this data. The classical models are refitted on the paper's training split
(same configuration as Table III); IndoBERT uses the three saved seed checkpoints. Overlap with our
corpus is measured first (TF-IDF cosine against every corpus document) and results are also given
on the articles with no corpus neighbour at cosine >= 0.8.
"""
import os, re, json
import numpy as np, pandas as pd, torch
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score, recall_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

S = os.path.dirname(os.path.abspath(__file__))
ext = pd.read_csv(os.path.join(S, "external", "mendeley", "600 news with valid hoax label.csv"), sep=";", encoding="cp1252")      # Windows-1252: curly quotes, dashes
ext["text"] = ext["berita"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
ye = (ext["tagging"].astype(str).str.strip().str.lower() == "hoax").astype(int).values

# the paper's classical preprocessing (rebuild_preproc.py)
STOP, st, cache = set(stopwords.words("indonesian")), StemmerFactory().create_stemmer(), {}
def prep(t):
    t = re.sub(r"[^a-zA-Z\s]", "", re.sub(r"http\S+|www\S+|https\S+", "", str(t).lower()))
    return " ".join(cache.setdefault(w, st.stem(w)) for w in t.split() if w not in STOP)
ext["clean"] = ext["text"].map(prep)

d = pd.read_csv(os.path.join(S, "deleaked2_preproc.csv")); X = d["text_clean"].astype(str).values; y = d["label"].values
itr, ite = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)

# overlap with the whole corpus
v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True); A = v.fit_transform(X)
near = np.asarray((v.transform(ext["clean"]) @ A.T).max(1).todense()).ravel()
clean_sub = near < 0.8

def scores(p, s, m):
    yt, p, s = ye[m], p[m], s[m]
    return {"n": int(m.sum()), "accuracy": accuracy_score(yt, p), "macro_f1": f1_score(yt, p, average="macro"),
            "balanced_accuracy": balanced_accuracy_score(yt, p), "roc_auc": roc_auc_score(yt, s),
            "recall_hoax": recall_score(yt, p, pos_label=1), "recall_valid": recall_score(yt, p, pos_label=0),
            "pred_hoax_rate": float(p.mean())}

res = {"dataset": "Pratiwi et al. 2017, Mendeley Data 10.17632/p3hfgr5j3m.1 (600 articles)", "n": len(ye),
       "hoax_share": float(ye.mean()), "median_words": int(ext["text"].str.split().str.len().median()),
       "overlap": {"max_cos_ge_0.8": int((near >= 0.8).sum()), "max_cos_ge_0.5": int((near >= 0.5).sum()),
                   "median_max_cos": float(np.median(near))}, "models": {}}
for name, mdl, ng in (("SVM", LinearSVC(C=1, random_state=42), (1, 3)), ("Logistic Regression", LogisticRegression(C=10, max_iter=1000, random_state=42), (1, 3))):
    vec = TfidfVectorizer(max_features=10000, ngram_range=ng); mdl.fit(vec.fit_transform(X[itr]), y[itr])
    B = vec.transform(ext["clean"]); p = mdl.predict(B); s = mdl.decision_function(B)
    res["models"][name] = {"all": scores(p, s, np.ones(len(ye), bool)), "no_overlap": scores(p, s, clean_sub)}

dev = "cuda" if torch.cuda.is_available() else "cpu"; ib = []
for seed in (42, 1, 2):
    path = os.path.join(S, "models", "ib_final_seed%d_best" % seed)
    tok = AutoTokenizer.from_pretrained(path); m = AutoModelForSequenceClassification.from_pretrained(path).to(dev).eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(ext), 32):
            enc = tok(list(ext["text"][i:i + 32]), truncation=True, max_length=128, padding=True, return_tensors="pt").to(dev)
            probs.append(torch.softmax(m(**enc).logits.float(), -1)[:, 1].cpu().numpy())
    pr = np.concatenate(probs); ib.append({"all": scores((pr > 0.5).astype(int), pr, np.ones(len(ye), bool)),
                                           "no_overlap": scores((pr > 0.5).astype(int), pr, clean_sub)})
res["models"]["IndoBERT"] = {k: {mt: float(np.mean([r[k][mt] for r in ib])) for mt in ib[0][k]} for k in ("all", "no_overlap")}
res["models"]["IndoBERT"]["sd_accuracy_all"] = float(np.std([r["all"]["accuracy"] for r in ib], ddof=1))
res["models"]["IndoBERT"]["per_seed_accuracy_all"] = [r["all"]["accuracy"] for r in ib]

json.dump(res, open(os.path.join(S, "out_external_pratiwi2017.json"), "w"), indent=2, default=float)
print(json.dumps({k: res[k] for k in ("n", "hoax_share", "median_words", "overlap")}, indent=1))
for k, r in res["models"].items():
    a = r["all"]; print("%-20s acc %.3f macroF1 %.3f balAcc %.3f AUC %.3f | hoax recall %.3f valid recall %.3f | no-overlap acc %.3f (n=%d)" % (
        k, a["accuracy"], a["macro_f1"], a["balanced_accuracy"], a["roc_auc"], a["recall_hoax"], a["recall_valid"], r["no_overlap"]["accuracy"], r["no_overlap"]["n"]))
