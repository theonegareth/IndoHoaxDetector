"""Fine-tune any HF encoder on the sanitised corpus.

Generalises indobert_retrain.py so the newer baselines (R1 #5) and the
minimal-preprocessing ablation (Q10) reuse one code path and one split.

  --text stemmed  text_clean          (same aggressive preprocessing as TF-IDF)
  --text minimal  title_dl + text_dl  (de-leaked, otherwise untouched)

Usage: train_transformer.py MODEL_ID TAG [--seed 42] [--text minimal] [--batch 16]
"""
import os, sys, json, time, random, argparse
import numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             confusion_matrix, classification_report)
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, Trainer,
                          TrainingArguments, DataCollatorWithPadding, set_seed)
from datasets import Dataset

S = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("tag")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--text", choices=["stemmed", "minimal"], default="minimal")
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--maxlen", type=int, default=128)
# XLM-R's 250k-token embedding matrix is 192M of its 278M parameters; with AdamW
# state it does not fit a 4 GB GPU. Freezing it keeps every transformer layer
# trainable. Recorded in the output so the deviation is always disclosed.
ap.add_argument("--freeze-embeddings", action="store_true")
a = ap.parse_args()

OUT = os.path.join(S, "hf_{}".format(a.tag))
RESULT = os.path.join(S, "out_hf_{}.json".format(a.tag))
if os.path.exists(RESULT):
    print("already done:", RESULT); sys.exit(0)

set_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
print("=== {} | {} | seed {} | {} text ===".format(a.tag, a.model, a.seed, a.text), flush=True)
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU", flush=True)

df = pd.read_csv(os.path.join(S, "deleaked2_preproc.csv"))
if a.text == "stemmed":
    txt = df["text_clean"].astype(str)
else:
    txt = (df["title_dl"].fillna("").astype(str) + " " + df["text_dl"].fillna("").astype(str)).str.strip()
X = np.asarray(txt.tolist(), dtype=object)
y = np.asarray(df["label"].tolist(), dtype=np.int64)
keep = np.array([len(s.split()) >= 3 for s in X])
X, y, idx = X[keep], y[keep], np.arange(len(y))[keep]

# same outer split as classical_sanitised.py; validation carved from train only
Xtr, Xte, ytr, yte, itr, ite = train_test_split(
    X, y, idx, test_size=0.2, random_state=42, stratify=y)
Xtr, Xva, ytr, yva = train_test_split(
    Xtr, ytr, test_size=0.1, random_state=a.seed, stratify=ytr)
print("train {} val {} test {}".format(len(Xtr), len(Xva), len(Xte)), flush=True)

tok = AutoTokenizer.from_pretrained(a.model)
t0 = time.time()
def mk(t, l):
    return Dataset.from_dict({"text": list(t), "labels": list(l)}).map(
        lambda b: tok(b["text"], truncation=True, max_length=a.maxlen),
        batched=True, remove_columns=["text"])
dtr, dva, dte = mk(Xtr, ytr), mk(Xva, yva), mk(Xte, yte)
tok_time = time.time() - t0
print("tokenisation {:.1f}s".format(tok_time), flush=True)

def metrics(p):
    pr = p.predictions.argmax(-1)
    return {"accuracy": accuracy_score(p.label_ids, pr),
            "f1": f1_score(p.label_ids, pr, average="weighted")}

model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=2)
if a.freeze_embeddings:
    for prm in model.base_model.embeddings.parameters():
        prm.requires_grad = False
n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_all_params = sum(p.numel() for p in model.parameters())
print("trainable params {:,} of {:,}{}".format(
    n_train_params, n_all_params, " (embeddings frozen)" if a.freeze_embeddings else ""), flush=True)
args = TrainingArguments(
    output_dir=OUT, num_train_epochs=a.epochs, per_device_train_batch_size=a.batch,
    per_device_eval_batch_size=64, learning_rate=a.lr, weight_decay=0.01,
    warmup_ratio=0.1, lr_scheduler_type="linear", seed=a.seed, data_seed=a.seed,
    eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
    metric_for_best_model="f1", greater_is_better=True, save_total_limit=1,
    logging_steps=500, fp16=torch.cuda.is_available(), report_to=[],
)
trainer = Trainer(model=model, args=args, train_dataset=dtr, eval_dataset=dva,
                  data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics)
t0 = time.time(); trainer.train(); train_time = time.time() - t0

p = trainer.predict(dte).predictions.argmax(-1)
cm = confusion_matrix(yte, p)
res = {
    "tag": a.tag, "model": a.model, "seed": a.seed, "text_variant": a.text,
    "max_length": a.maxlen, "batch_size": a.batch, "epochs": a.epochs,
    "learning_rate": a.lr, "weight_decay": 0.01, "warmup_ratio": 0.1,
    "scheduler": "linear", "optimizer": "adamw_torch", "fp16": bool(torch.cuda.is_available()),
    "early_stopping": "load_best_model_at_end on validation weighted F1",
    "frozen_embeddings": bool(a.freeze_embeddings),
    "trainable_params": int(n_train_params), "total_params": int(n_all_params),
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    "n_train": len(Xtr), "n_val": len(Xva), "n_test": len(Xte),
    "tokenisation_time_s": round(tok_time, 1), "training_time_s": round(train_time, 1),
    "accuracy": float(accuracy_score(yte, p)),
    "precision_weighted": float(precision_score(yte, p, average="weighted", zero_division=0)),
    "recall_weighted": float(recall_score(yte, p, average="weighted", zero_division=0)),
    "f1_weighted": float(f1_score(yte, p, average="weighted", zero_division=0)),
    "per_class": classification_report(yte, p, target_names=["factual", "hoax"],
                                       output_dict=True, zero_division=0),
    "confusion_matrix": cm.tolist(),
}
src = np.asarray(df["source"].astype(str).tolist(), dtype=object)[ite]
res["per_source_accuracy"] = {s: {"n": int((src == s).sum()),
                                  "accuracy": float(accuracy_score(yte[src == s], p[src == s]))}
                              for s in np.unique(src)}
np.savez(os.path.join(S, "out_hf_{}.npz".format(a.tag)), pred=p, true=yte, test_idx=ite)
with open(RESULT, "w") as fh:
    json.dump(res, fh, indent=2)
print("### {}: accuracy {:.4f}  weighted F1 {:.4f}  ({:.0f}s)".format(
    a.tag, res["accuracy"], res["f1_weighted"], train_time), flush=True)
print("wrote", RESULT, flush=True)
sys.stdout.flush(); sys.stderr.flush()

# Hard exit. Interpreter teardown can hang here holding the GPU, which stalls
# the rest of the queue; the results are already on disk and the OS reclaims
# VRAM on exit.
os._exit(0)
