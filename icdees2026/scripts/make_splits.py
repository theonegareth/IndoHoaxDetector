"""Export every evaluation split as row indices, so IndoBERT runs anywhere on exactly the splits the SVM used.

Splits (indices into deleaked2_preproc.csv; IndoBERT's >=3-word filter is applied later, at training time):
  random        80/20 stratified, seed 42                      (headline split, train_ib_final.py)
  xsource       train CNN+Kompas+TurnBackHoax, test Detik+Tempo+Komdigi        (cross_source.py B)
  loso_<src>    train every other source, test <src>                           (cross_source.py C)
  topic         12 k-means TF-IDF clusters, whole clusters held out (~20%)     (topic_split.py)
  topic_ctl     random stratified split of the same size                       (topic_split.py)
  time          train before 2023-01, test from 2023-01, only the two collections
                spanning the cutoff (cnn_10k_cleaned, turnbackhoax_cleaned)    (paper Table V "same collections")
  time_ctl      random stratified split of that same subset and size
Each split is checked against the SVM numbers already in the paper before it is saved.
"""
import os, re, json
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

S = os.path.dirname(os.path.abspath(__file__)); SEED = 42
df = pd.read_csv(os.path.join(S, "deleaked2_preproc.csv"))
df["text_clean"] = df["text_clean"].astype(str); df["url"] = df["url"].astype(str)
idx, y = np.arange(len(df)), df["label"].values
splits = {}

def svm_f1(tr, te):
    v = TfidfVectorizer(max_features=10000, ngram_range=(1, 2)); A = v.fit_transform(df.text_clean.values[tr])
    p = LinearSVC(C=1.0, random_state=SEED).fit(A, y[tr]).predict(v.transform(df.text_clean.values[te]))
    return f1_score(y[te], p, average="weighted")

tr, te = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y); splits["random"] = (tr, te)

TR = ["cnn_10k_cleaned", "cnn_cleaned", "kompas_4k_cleaned", "kompas_cleaned", "turnbackhoax_cleaned", "turnbackhoax_10k_cleaned"]
m = df.source.isin(TR).values; splits["xsource"] = (idx[m], idx[~m])
for s in sorted(df.source.unique()):
    m = (df.source == s).values; splits["loso_" + s] = (idx[~m], idx[m])

# topic split: identical construction to topic_split.py
Xt = TfidfVectorizer(max_features=20000, ngram_range=(1, 1), min_df=3).fit_transform(df.text_clean)
topic = MiniBatchKMeans(n_clusters=12, random_state=SEED, n_init=5, batch_size=2048).fit_predict(Xt)
sizes = pd.Series(topic).value_counts().sort_index()      # by cluster id, as groupby() orders it in topic_split.py
held, cum = [], 0
for t in sizes.sample(frac=1.0, random_state=SEED).index:
    if cum >= 0.2 * len(df): break
    held.append(t); cum += sizes[t]
m = np.isin(topic, held); splits["topic"] = (idx[~m], idx[m])
splits["topic_ctl"] = tuple(train_test_split(idx, test_size=m.sum() / len(df), random_state=SEED, stratify=y))

# time split within the two collections that span the cutoff
PATS = [r"/(20\d{2})/(\d{2})/(\d{2})/", r"/(20\d{2})/(\d{2})/", r"/(20\d{2})(\d{2})(\d{2})\d{6}-"]
def ym(u):
    for p in PATS:
        g = re.search(p, u)
        if g:
            yy, mo = int(g.group(1)), int(g.group(2))
            if 2010 <= yy <= 2026 and 1 <= mo <= 12: return yy * 100 + mo
    return np.nan
d = df.url.map(ym).values
sub = idx[df.source.isin(["cnn_10k_cleaned", "turnbackhoax_cleaned"]).values & ~np.isnan(d)]
splits["time"] = (sub[d[sub] < 202301], sub[d[sub] >= 202301])
splits["time_ctl"] = tuple(train_test_split(sub, test_size=len(splits["time"][1]) / len(sub), random_state=SEED, stratify=y[sub]))

# checks against the published SVM results
ref = {"xsource": 0.8842, "topic": 0.9654, "topic_ctl": 0.9756, "time": 0.9632, "time_ctl": 0.9930}
for k, want in ref.items():
    got = svm_f1(*splits[k]); print(f"{k:10} n_train {len(splits[k][0]):6} n_test {len(splits[k][1]):6}  SVM wF1 {got:.4f} (paper {want})")
    assert abs(got - want) < 0.0006, (k, got, want)
assert (len(splits["time"][0]), len(splits["time"][1])) == (14637, 7562)

np.savez_compressed(os.path.join(S, "splits.npz"), **{k + "__tr": a for k, (a, b) in splits.items()}, **{k + "__te": b for k, (a, b) in splits.items()})
mini = (df.title_dl.fillna("").astype(str) + " " + df.text_dl.fillna("").astype(str)).str.strip()
pd.DataFrame({"text": mini, "label": df.label, "source": df.source}).to_csv(os.path.join(S, "pod_corpus.csv.gz"), index=False, compression="gzip")
print("wrote splits.npz (%d splits) and pod_corpus.csv.gz" % len(splits))
