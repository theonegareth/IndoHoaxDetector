"""ICDEES R2 §2-§4: full metric set + efficiency for the four classical models.

Refits each model at the configuration selected by 5-fold CV in classical_sanitised.py
(read from out_classical_sanitised.json, so nothing is retyped), on the identical
80/20 split, and records:
  accuracy, weighted/macro F1, balanced accuracy, ROC-AUC, bootstrap 95% CIs,
  training time (vectorise + fit), inference time, model size on disk, peak memory.
Saves predictions + scores for the error analysis and McNemar re-checks.
"""
import os, json, time, pickle, tracemalloc
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score,
                             precision_score, recall_score, confusion_matrix)

S = os.path.dirname(os.path.abspath(__file__))
SEED, NBOOT = 42, 2000
cfg = json.load(open(os.path.join(S, "out_classical_sanitised.json")))["results"]

df = pd.read_csv(os.path.join(S, "deleaked2_preproc.csv"))
X = np.asarray(df["text_clean"].astype(str).tolist(), dtype=object)
y = np.asarray(df["label"].tolist(), dtype=np.int64)
Xtr, Xte, ytr, yte, itr, ite = train_test_split(
    X, y, np.arange(len(y)), test_size=0.2, random_state=SEED, stratify=y)


def make(name, params):
    v = float(params.split("=")[1]) if "=" in params else None
    return {"Logistic Regression": lambda: LogisticRegression(C=v, max_iter=1000, random_state=SEED),
            "SVM": lambda: LinearSVC(C=v, random_state=SEED),
            "Naive Bayes": lambda: MultinomialNB(alpha=v),
            "Random Forest": lambda: RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)}[name]()


def boot_ci(y_true, y_pred, score, fn, rng):
    n = len(y_true); vals = []
    for _ in range(NBOOT):
        b = rng.integers(0, n, n)
        vals.append(fn(y_true[b], y_pred[b] if score is None else score[b]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


out, saved = {}, {"y_true": yte, "test_idx": ite}
for name, r in cfg.items():
    mf, ng = r["best_tfidf_max_features"], tuple(r["best_tfidf_ngram"])
    # timed pass (no tracing, so the reported times are clean)
    t0 = time.perf_counter(); vec = TfidfVectorizer(max_features=mf, ngram_range=ng)
    A = vec.fit_transform(Xtr); t_vec = time.perf_counter() - t0
    t0 = time.perf_counter(); m = make(name, r["best_model_params"]); m.fit(A, ytr); t_fit = time.perf_counter() - t0
    # separate memory pass: tracemalloc slows allocation, so it must not overlap the timing
    tracemalloc.start()
    _v = TfidfVectorizer(max_features=mf, ngram_range=ng); _m = make(name, r["best_model_params"])
    _m.fit(_v.fit_transform(Xtr), ytr)
    _, peak_train = tracemalloc.get_traced_memory(); tracemalloc.stop()
    del _v, _m

    # inference: raw preprocessed text in -> label out, timed over the whole test set
    t0 = time.perf_counter(); B = vec.transform(Xte); p = m.predict(B); t_inf = time.perf_counter() - t0
    score = m.decision_function(B) if hasattr(m, "decision_function") else m.predict_proba(B)[:, 1]
    size = len(pickle.dumps(vec)) + len(pickle.dumps(m))

    rng = np.random.default_rng(SEED)
    res = {
        "config": "TF-IDF {} {}-grams, {}".format(mf, ng, r["best_model_params"]),
        "svm_kernel": "linear (LinearSVC, liblinear)" if name == "SVM" else None,
        "accuracy": accuracy_score(yte, p),
        "precision_weighted": precision_score(yte, p, average="weighted"),
        "recall_weighted": recall_score(yte, p, average="weighted"),
        "f1_weighted": f1_score(yte, p, average="weighted"),
        "f1_macro": f1_score(yte, p, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(yte, p),
        "roc_auc": roc_auc_score(yte, score),
        "ci95_accuracy": boot_ci(yte, p, None, accuracy_score, rng),
        "ci95_f1_macro": boot_ci(yte, p, None, lambda a, b: f1_score(a, b, average="macro"), rng),
        "ci95_roc_auc": boot_ci(yte, p, score, roc_auc_score, rng),
        "confusion_matrix": confusion_matrix(yte, p).tolist(),
        "train_time_s": t_vec + t_fit, "vectorise_time_s": t_vec, "fit_time_s": t_fit,
        "inference_time_s_test_set": t_inf, "n_test": int(len(yte)),
        "inference_ms_per_doc": 1000 * t_inf / len(yte),
        "model_size_mb": size / 1e6, "peak_python_memory_train_mb": peak_train / 1e6,
    }
    out[name] = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in res.items()}
    saved[name.replace(" ", "_") + "_pred"] = p
    saved[name.replace(" ", "_") + "_score"] = score
    # cross-check against the CV run this config came from
    assert abs(res["f1_weighted"] - r["test_f1_weighted"]) < 1e-9, (name, res["f1_weighted"], r["test_f1_weighted"])
    print("{:20} acc {:.4f} [{:.4f},{:.4f}]  macroF1 {:.4f}  balAcc {:.4f}  AUC {:.4f}  "
          "train {:.1f}s  infer {:.3f} ms/doc  size {:.1f} MB  peak {:.0f} MB".format(
              name, res["accuracy"], *res["ci95_accuracy"], res["f1_macro"], res["balanced_accuracy"],
              res["roc_auc"], res["train_time_s"], res["inference_ms_per_doc"], res["model_size_mb"],
              res["peak_python_memory_train_mb"]), flush=True)

np.savez(os.path.join(S, "out_classical_preds_scores.npz"), **saved)
with open(os.path.join(S, "out_metrics_classical.json"), "w") as fh:
    json.dump(out, fh, indent=2)
print("wrote out_metrics_classical.json  (all weighted F1 values match the CV run exactly)")
