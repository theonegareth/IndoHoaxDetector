"""Headline IndoBERT runs for the ICDEES camera-ready (R2 §2, §3, §4).

Same configuration as the revision's headline model (minimal preprocessing, lr 1e-5,
3 epochs, batch 16, max_len 128, fp16, best epoch on a 10% validation split), plus
everything the reviewer asked for that the earlier runs did not record:
  * class probabilities  -> ROC-AUC, macro-F1, balanced accuracy, bootstrap CIs
  * efficiency           -> GPU and CPU inference latency, model size, peak GPU memory
  * energy               -> GPU board power sampled once a second via nvidia-smi
The checkpoint is kept (persistent folder) so the models can be published.

Usage: train_ib_final.py SEED
"""
import os, sys, json, time, random, threading, subprocess
import numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score,
                             precision_score, recall_score, confusion_matrix)
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, Trainer,
                          TrainingArguments, DataCollatorWithPadding, set_seed)

S = os.path.dirname(os.path.abspath(__file__))
SEED = int(sys.argv[1]); MODEL = "indobenchmark/indobert-base-p1"
XSOURCE = len(sys.argv) > 2 and sys.argv[2] == "xsource"   # publisher-disjoint split, as in cross_source.py
TAG = "ib_final_seed{}".format(SEED) + ("_xsource" if XSOURCE else "")
SMOKE = bool(os.environ.get("SMOKE")); TAG += "_smoke" if SMOKE else ""   # quick end-to-end check on 2,000 docs
RESULT = os.path.join(S, "out_{}.json".format(TAG))
if os.path.exists(RESULT):
    print("already done:", RESULT); os._exit(0)
NVSMI = r"C:\Windows\System32\nvidia-smi.exe"


class PowerMeter:
    """Integrates GPU board power (W) over wall time -> Wh. nvidia-smi reports board power."""
    def __init__(self): self.samples, self._stop = [], threading.Event()
    def _run(self):
        while not self._stop.is_set():
            try:
                w = float(subprocess.check_output([NVSMI, "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                                                  timeout=5).decode().strip().splitlines()[0])
                self.samples.append((time.time(), w))
            except Exception:
                pass
            self._stop.wait(1.0)
    def __enter__(self):
        self.t = threading.Thread(target=self._run, daemon=True); self.t.start(); return self
    def __exit__(self, *a):
        self._stop.set(); self.t.join()
    def wh(self):
        if len(self.samples) < 2: return float("nan")
        ts, ws = np.array([s[0] for s in self.samples]), np.array([s[1] for s in self.samples])
        return float(np.trapezoid(ws, ts) / 3600.0)
    def mean_w(self):
        return float(np.mean([s[1] for s in self.samples])) if self.samples else float("nan")


set_seed(SEED); random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print("=== {} | {} | {} ===".format(TAG, MODEL, dev), flush=True)

df = pd.read_csv(os.path.join(S, "deleaked2_preproc.csv"))
txt = (df["title_dl"].fillna("").astype(str) + " " + df["text_dl"].fillna("").astype(str)).str.strip()
X = np.asarray(txt.tolist(), dtype=object); y = np.asarray(df["label"].tolist(), dtype=np.int64)
keep = np.array([len(s.split()) >= 3 for s in X]); X, y, idx = X[keep], y[keep], np.arange(len(y))[keep]
if SMOKE: X, y, idx = X[::30], y[::30], idx[::30]
if XSOURCE:
    TR = ["cnn_10k_cleaned", "cnn_cleaned", "kompas_4k_cleaned", "kompas_cleaned",
          "turnbackhoax_cleaned", "turnbackhoax_10k_cleaned"]          # test: Detik, Tempo, Komdigi
    m = np.isin(df["source"].values[idx], TR)
    Xtr, Xte, ytr, yte, itr, ite = X[m], X[~m], y[m], y[~m], idx[m], idx[~m]
else:
    Xtr, Xte, ytr, yte, itr, ite = train_test_split(X, y, idx, test_size=0.2, random_state=42, stratify=y)
Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=0.1, random_state=SEED, stratify=ytr)

tok = AutoTokenizer.from_pretrained(MODEL)
def mk(t, l):   # a list of feature dicts is a valid Trainer dataset; avoids the `datasets` package
    enc = tok(list(t), truncation=True, max_length=128)
    return [{**{k: enc[k][i] for k in enc}, "labels": int(l[i])} for i in range(len(l))]
t0 = time.time(); dtr, dva, dte = mk(Xtr, ytr), mk(Xva, yva), mk(Xte, yte); tok_time = time.time() - t0

model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
OUT = os.path.join(S, "models", TAG)
args = TrainingArguments(
    output_dir=OUT, num_train_epochs=3, per_device_train_batch_size=16, per_device_eval_batch_size=64,
    learning_rate=1e-5, weight_decay=0.01, warmup_ratio=0.1, lr_scheduler_type="linear",
    seed=SEED, data_seed=SEED, eval_strategy="epoch", save_strategy="epoch",
    load_best_model_at_end=True, metric_for_best_model="f1", greater_is_better=True,
    save_total_limit=1, logging_steps=500, fp16=torch.cuda.is_available(), report_to=[])
metrics = lambda p: {"f1": f1_score(p.label_ids, p.predictions.argmax(-1), average="weighted")}
trainer = Trainer(model=model, args=args, train_dataset=dtr, eval_dataset=dva,
                  data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics)

torch.cuda.reset_peak_memory_stats()
with PowerMeter() as pm_train:
    t0 = time.time(); trainer.train(); train_time = time.time() - t0
peak_train_gb = torch.cuda.max_memory_allocated() / 1e9

# GPU inference over the full test set (batch 64), timed and metered
torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
with PowerMeter() as pm_inf:
    t0 = time.time(); logits = trainer.predict(dte).predictions; torch.cuda.synchronize(); inf_time = time.time() - t0
peak_inf_gb = torch.cuda.max_memory_allocated() / 1e9
prob = torch.softmax(torch.tensor(logits, dtype=torch.float32), -1)[:, 1].numpy(); pred = logits.argmax(-1)

# save the best checkpoint as a clean model folder for publication
final_dir = os.path.join(S, "models", TAG + "_best"); trainer.save_model(final_dir); tok.save_pretrained(final_dir)
size_mb = sum(os.path.getsize(os.path.join(final_dir, f)) for f in os.listdir(final_dir)) / 1e6
n_params = sum(p.numel() for p in trainer.model.parameters())

# CPU inference latency on 1,000 test documents, batch 1 (a deployment-style number)
cpu_model = AutoModelForSequenceClassification.from_pretrained(final_dir).eval()
torch.set_num_threads(os.cpu_count())
sub = list(Xte[:1000]); t0 = time.time()
with torch.no_grad():
    for s in sub:
        cpu_model(**tok(s, truncation=True, max_length=128, return_tensors="pt"))
cpu_ms = 1000 * (time.time() - t0) / len(sub)

rng = np.random.default_rng(SEED)
def ci(fn, use_prob=False):
    v = []
    for _ in range(2000):
        b = rng.integers(0, len(yte), len(yte)); v.append(fn(yte[b], (prob if use_prob else pred)[b]))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

res = {
    "tag": TAG, "seed": SEED, "model": MODEL, "device": dev, "text_variant": "minimal",
    "n_train": len(Xtr), "n_val": len(Xva), "n_test": len(Xte), "n_params": int(n_params),
    "accuracy": float(accuracy_score(yte, pred)),
    "precision_weighted": float(precision_score(yte, pred, average="weighted")),
    "recall_weighted": float(recall_score(yte, pred, average="weighted")),
    "f1_weighted": float(f1_score(yte, pred, average="weighted")),
    "f1_macro": float(f1_score(yte, pred, average="macro")),
    "balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
    "roc_auc": float(roc_auc_score(yte, prob)),
    "ci95_accuracy": ci(accuracy_score), "ci95_f1_macro": ci(lambda a, b: f1_score(a, b, average="macro")),
    "ci95_roc_auc": ci(roc_auc_score, use_prob=True),
    "confusion_matrix": confusion_matrix(yte, pred).tolist(),
    "tokenisation_time_s": tok_time, "train_time_s": train_time,
    "gpu_inference_time_s_test_set": inf_time, "gpu_inference_ms_per_doc": 1000 * inf_time / len(yte),
    "cpu_inference_ms_per_doc_batch1": cpu_ms,
    "model_size_mb": size_mb, "peak_gpu_memory_train_gb": peak_train_gb, "peak_gpu_memory_infer_gb": peak_inf_gb,
    "gpu_energy_train_wh": pm_train.wh(), "gpu_mean_power_train_w": pm_train.mean_w(),
    "gpu_energy_infer_wh": pm_inf.wh(), "power_samples_train": len(pm_train.samples),
}
np.savez(os.path.join(S, "out_{}.npz".format(TAG)), pred=pred, prob=prob, true=yte, test_idx=ite)
with open(RESULT, "w") as fh:
    json.dump(res, fh, indent=2)
print("### {}: acc {:.4f} macroF1 {:.4f} AUC {:.4f} | train {:.0f}s {:.1f} Wh | GPU {:.2f} ms/doc | CPU {:.1f} ms/doc | {:.0f} MB".format(
    TAG, res["accuracy"], res["f1_macro"], res["roc_auc"], train_time, res["gpu_energy_train_wh"],
    res["gpu_inference_ms_per_doc"], cpu_ms, size_mb), flush=True)
sys.stdout.flush(); os._exit(0)
