"""Topic-separated split (the remaining option in R2 Q6).

Topics are unlabelled here, so they are induced with k-means over TF-IDF and
whole clusters are held out. A model that has learned topical shortcuts rather
than hoax cues degrades when the test topics are unseen.
"""
import os, json
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

S = os.path.dirname(os.path.abspath(__file__))
SEED, K = 42, 12
df = pd.read_csv(os.path.join(S, "deleaked2_preproc.csv"))
df["text_clean"] = df["text_clean"].astype(str)

vt = TfidfVectorizer(max_features=20000, ngram_range=(1, 1), min_df=3)
Xt = vt.fit_transform(df["text_clean"])
km = MiniBatchKMeans(n_clusters=K, random_state=SEED, n_init=5, batch_size=2048)
df["topic"] = km.fit_predict(Xt)

sizes = df.groupby("topic").agg(n=("label", "size"), hoax_rate=("label", "mean"))
print("induced topics (k={}):".format(K))
print(sizes.to_string())

# hold out whole clusters, ~20% of documents, keeping both classes in the test set
order = sizes.sample(frac=1.0, random_state=SEED).index.tolist()
held, cum = [], 0
for t in order:
    if cum >= 0.2 * len(df):
        break
    held.append(t); cum += sizes.loc[t, "n"]
tr, te = df[~df.topic.isin(held)], df[df.topic.isin(held)]
print("\nheld-out topics {} -> test {:,} ({:.1%}), train {:,}".format(
    held, len(te), len(te) / len(df), len(tr)))
print("  test labels", te.label.value_counts().to_dict())

res = {}
if te["label"].nunique() < 2:
    print("held-out topics are single-class; cannot evaluate")
else:
    v = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    A = v.fit_transform(tr["text_clean"])
    m = LinearSVC(C=1.0, random_state=SEED).fit(A, tr["label"])
    p = m.predict(v.transform(te["text_clean"]))
    y = te["label"].values
    res["topic_separated"] = {
        "k": K, "held_out_topics": [int(t) for t in held],
        "n_train": len(tr), "n_test": len(te),
        "accuracy": float(accuracy_score(y, p)),
        "f1_weighted": float(f1_score(y, p, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y, p, labels=[0, 1]).tolist(),
    }
    print("  TOPIC-SEPARATED  acc {:.4f}  F1 {:.4f}".format(
        res["topic_separated"]["accuracy"], res["topic_separated"]["f1_weighted"]))

    # like-for-like random control of the same size
    rtr, rte = train_test_split(df, test_size=len(te) / len(df), random_state=SEED,
                                stratify=df["label"])
    v2 = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    A2 = v2.fit_transform(rtr["text_clean"])
    m2 = LinearSVC(C=1.0, random_state=SEED).fit(A2, rtr["label"])
    p2 = m2.predict(v2.transform(rte["text_clean"]))
    res["random_control"] = {
        "accuracy": float(accuracy_score(rte["label"], p2)),
        "f1_weighted": float(f1_score(rte["label"], p2, average="weighted", zero_division=0)),
    }
    print("  RANDOM control    acc {:.4f}  F1 {:.4f}".format(
        res["random_control"]["accuracy"], res["random_control"]["f1_weighted"]))

with open(os.path.join(S, "out_topic_split.json"), "w") as fh:
    json.dump(res, fh, indent=2)
print("\nwrote out_topic_split.json")
