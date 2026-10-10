"""Measured CPU energy for the classical models (replaces the 115 W upper bound in Table IV).

Reads CPU package power from LibreHardwareMonitor's web server (Options > Remote Web Server > Run,
default http://localhost:8085/data.json; LibreHardwareMonitor must run as administrator).
Power is sampled every 0.5 s during each model's training (TF-IDF vectorisation + fit, the same
configuration as metrics_classical.py), integrated to Wh, and an idle baseline measured first is
also reported so energy above idle can be separated from the laptop's background draw.
"""
import os, json, time, threading, urllib.request
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

URL = "http://localhost:8085/data.json"
S = os.path.dirname(os.path.abspath(__file__))

def package_watts():
    def walk(n, parent=""):
        if n.get("Text") == "CPU Package" and parent == "Powers":
            return float(str(n["Value"]).split()[0].replace(",", "."))
        for c in n.get("Children", []):
            v = walk(c, n.get("Text", ""))
            if v is not None: return v
    return walk(json.load(urllib.request.urlopen(URL, timeout=2)))

class Meter:
    def __init__(self): self.s, self.stop = [], threading.Event()
    def _run(self):
        while not self.stop.is_set():
            try: self.s.append((time.time(), package_watts()))
            except Exception: pass
            self.stop.wait(0.5)
    def __enter__(self): self.t = threading.Thread(target=self._run, daemon=True); self.t.start(); return self
    def __exit__(self, *a): self.stop.set(); self.t.join()
    def wh(self):
        t, w = np.array([x[0] for x in self.s]), np.array([x[1] for x in self.s]); return float(np.trapezoid(w, t) / 3600)
    def mean_w(self): return float(np.mean([x[1] for x in self.s]))

if __name__ == "__main__":
    print("CPU package now: %.1f W" % package_watts())          # fails fast if the web server is not running
    with Meter() as idle: time.sleep(30)
    idle_w = idle.mean_w(); print("idle baseline %.1f W" % idle_w)
    cfg = json.load(open(os.path.join(S, "out_classical_sanitised.json")))["results"]
    df = pd.read_csv(os.path.join(S, "deleaked2_preproc.csv")); X = df["text_clean"].astype(str).values; y = df["label"].values
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    mk = {"Logistic Regression": lambda v: LogisticRegression(C=v, max_iter=1000, random_state=42),
          "SVM": lambda v: LinearSVC(C=v, random_state=42), "Naive Bayes": lambda v: MultinomialNB(alpha=v),
          "Random Forest": lambda v: RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)}
    out = {"idle_w": idle_w, "models": {}}
    for name, r in cfg.items():
        v = float(r["best_model_params"].split("=")[1]) if "=" in r["best_model_params"] else None
        with Meter() as m:
            t0 = time.time(); vec = TfidfVectorizer(max_features=r["best_tfidf_max_features"], ngram_range=tuple(r["best_tfidf_ngram"]))
            mk[name](v).fit(vec.fit_transform(Xtr), ytr); dt = time.time() - t0
        out["models"][name] = {"train_s": dt, "energy_wh": m.wh(), "mean_w": m.mean_w(),
                               "energy_above_idle_wh": m.wh() - idle_w * dt / 3600, "samples": len(m.s)}
        print("%-20s %.1f s  %.3f Wh total  %.3f Wh above idle  (mean %.1f W)" % (name, dt, m.wh(), out["models"][name]["energy_above_idle_wh"], m.mean_w()))
    json.dump(out, open(os.path.join(S, "out_energy_classical.json"), "w"), indent=2)
