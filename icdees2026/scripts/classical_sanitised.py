"""Tuned classical benchmark on the SANITISED (de-leaked) corpus.

Same protocol as classical.py so the two tables are directly comparable:
80/20 stratified split (seed 42), 5-fold CV over the TF-IDF grid crossed with
each model's own grid, per-fold scores retained for paired testing.

The split here is the one indobert_retrain.py also uses, so IndoBERT can be
compared against these models on identical data.
"""
import os, json, time
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (f1_score, accuracy_score, precision_score, recall_score,
                             confusion_matrix, classification_report)

S = os.path.dirname(os.path.abspath(__file__))
SEED = 42
df = pd.read_csv(os.path.join(S, "deleaked2_preproc.csv"))
df["text_clean"] = df["text_clean"].astype(str)
X, y = df["text_clean"].values, df["label"].values

Xtr, Xte, ytr, yte, itr, ite = train_test_split(
    X, y, np.arange(len(y)), test_size=0.2, random_state=SEED, stratify=y)
print("SANITISED corpus: {} docs, train {}, test {}".format(len(df), len(Xtr), len(Xte)), flush=True)

TFIDF_GRID = [(mf, ng) for mf in (1000, 3000, 5000, 10000) for ng in ((1, 1), (1, 2), (1, 3))]
MODELS = {
    "Logistic Regression": [("C=%g" % c, (lambda c=c: LogisticRegression(C=c, max_iter=1000, random_state=SEED))) for c in (0.1, 1, 10)],
    "SVM":                 [("C=%g" % c, (lambda c=c: LinearSVC(C=c, random_state=SEED))) for c in (0.1, 1, 10)],
    "Naive Bayes":         [("alpha=%g" % a, (lambda a=a: MultinomialNB(alpha=a))) for a in (0.01, 0.1, 1.0)],
    "Random Forest":       [("n=200", (lambda: RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)))],
}

folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(Xtr, ytr))
fold_scores = {m: {} for m in MODELS}
t0 = time.time()

for mf, ng in TFIDF_GRID:
    key_tf = "mf{}_ng{}{}".format(mf, ng[0], ng[1])
    acc = {m: {c: [] for c, _ in cfgs} for m, cfgs in MODELS.items()}
    for tr, va in folds:
        v = TfidfVectorizer(max_features=mf, ngram_range=ng)
        A, B = v.fit_transform(Xtr[tr]), None
        B = v.transform(Xtr[va])
        for mname, cfgs in MODELS.items():
            for cname, ctor in cfgs:
                m = ctor(); m.fit(A, ytr[tr])
                acc[mname][cname].append(f1_score(ytr[va], m.predict(B), average="weighted"))
    for mname in MODELS:
        for cname, sc in acc[mname].items():
            fold_scores[mname][key_tf + "|" + cname] = sc
    print("  {} done  {:.0f}s".format(key_tf, time.time() - t0), flush=True)

results, preds = {}, {}
for mname, cfgs in MODELS.items():
    best = max(fold_scores[mname], key=lambda k: np.mean(fold_scores[mname][k]))
    tf_part, cname = best.split("|")
    mf = int(tf_part.split("_")[0][2:])
    ng = (int(tf_part.split("ng")[1][0]), int(tf_part.split("ng")[1][1]))
    ctor = dict(cfgs)[cname]

    t = time.time(); v = TfidfVectorizer(max_features=mf, ngram_range=ng)
    A = v.fit_transform(Xtr); vec_t = time.time() - t
    t = time.time(); m = ctor(); m.fit(A, ytr); fit_t = time.time() - t
    p = m.predict(v.transform(Xte)); preds[mname] = p

    results[mname] = {
        "best_tfidf_max_features": mf, "best_tfidf_ngram": list(ng), "best_model_params": cname,
        "cv_f1_mean": float(np.mean(fold_scores[mname][best])),
        "cv_f1_std": float(np.std(fold_scores[mname][best], ddof=1)),
        "cv_fold_f1": [float(x) for x in fold_scores[mname][best]],
        "test_accuracy": float(accuracy_score(yte, p)),
        "test_precision_weighted": float(precision_score(yte, p, average="weighted", zero_division=0)),
        "test_recall_weighted": float(recall_score(yte, p, average="weighted", zero_division=0)),
        "test_f1_weighted": float(f1_score(yte, p, average="weighted", zero_division=0)),
        "per_class": classification_report(yte, p, target_names=["factual", "hoax"],
                                           output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(yte, p).tolist(),
        "fit_time_s": round(fit_t, 2), "vectorize_time_s": round(vec_t, 2),
        "total_train_time_s": round(fit_t + vec_t, 2),
    }
    r = results[mname]
    print("\n### {}  [{} {}]".format(mname, tf_part, cname))
    print("  CV F1 {:.4f} +- {:.4f}   TEST F1 {:.4f}  acc {:.4f}".format(
        r["cv_f1_mean"], r["cv_f1_std"], r["test_f1_weighted"], r["test_accuracy"]))
    print("  confusion:", r["confusion_matrix"], flush=True)

np.savez(os.path.join(S, "out_sanitised_preds.npz"), y_true=yte, test_idx=ite,
         **{k.replace(" ", "_"): v for k, v in preds.items()})
json.dump({"seed": SEED, "corpus": "deleaked2_preproc.csv",
           "n_train": len(Xtr), "n_test": len(Xte), "results": results,
           "all_fold_scores": {m: {k: [float(x) for x in v] for k, v in d.items()}
                               for m, d in fold_scores.items()}},
          open(os.path.join(S, "out_classical_sanitised.json"), "w"), indent=2)
print("\nDONE {:.0f}s -> out_classical_sanitised.json".format(time.time() - t0))
