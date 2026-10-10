"""Grid-edge check: LogReg C=10 and NB alpha=0.01 were at the edge of their grids.
Extend each one step with the same 5-fold CV on the training split only (TF-IDF 10k, best n-grams)."""
import json, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
d = pd.read_csv("deleaked2_preproc.csv"); X = d.text_clean.astype(str).values; y = d.label.values
Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
cv = StratifiedKFold(5, shuffle=True, random_state=42); out = {}
for name, ng, vals, mk in (("Logistic Regression", (1, 3), (10, 30, 100), lambda c: LogisticRegression(C=c, max_iter=2000, random_state=42)),
                           ("Naive Bayes", (1, 2), (0.01, 0.003, 0.001), lambda a: MultinomialNB(alpha=a))):
    for v in vals:
        s = cross_val_score(make_pipeline(TfidfVectorizer(max_features=10000, ngram_range=ng), mk(v)), Xtr, ytr, cv=cv, scoring="f1_weighted", n_jobs=5)
        out[f"{name} {v}"] = [float(s.mean()), float(s.std())]; print(name, v, round(100 * s.mean(), 2), "±", round(100 * s.std(), 2), flush=True)
json.dump(out, open("out_grid_edge_check.json", "w"), indent=2)
