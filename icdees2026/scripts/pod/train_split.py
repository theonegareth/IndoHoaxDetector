"""Fine-tune one encoder on one exported split (splits.npz) with the paper's IndoBERT settings.

Same configuration as train_ib_final.py: minimal (unstemmed) text, max length 128, batch 16,
3 epochs, AdamW, weight decay 0.01, 10% warm-up, linear decay, fp16, best epoch chosen on a
10% stratified validation split of the training indices. The test indices are never used for
selection. Writes results/out_pod_<tag>.json and .npz (pred, prob, true, test_idx).

Usage: train_split.py TAG SPLIT SEED LR MODEL
"""
import os, sys, json, time, random, shutil
import numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score, recall_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments,
                          DataCollatorWithPadding, set_seed)

TAG, SPLIT, SEED, LR, MODEL = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
W = os.path.dirname(os.path.abspath(__file__)); OUTD = os.path.join(W, "results"); os.makedirs(OUTD, exist_ok=True)
RESULT = os.path.join(OUTD, "out_pod_%s.json" % TAG)
if os.path.exists(RESULT):
    print("already done:", TAG); os._exit(0)
set_seed(SEED); random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

df = pd.read_csv(os.path.join(W, "pod_corpus.csv.gz"))
X = df["text"].fillna("").astype(str).values; y = df["label"].values.astype(np.int64)
sp = np.load(os.path.join(W, "splits.npz"))
keep = np.array([len(s.split()) >= 3 for s in X])              # same filter as train_ib_final.py
tr, te = sp[SPLIT + "__tr"], sp[SPLIT + "__te"]
tr, te = tr[keep[tr]], te[keep[te]]
if SPLIT == "random": assert len(te) == 12587, len(te)            # identical to the paper's headline split
trn, val = train_test_split(tr, test_size=0.1, random_state=SEED, stratify=y[tr])
if os.environ.get("SMOKE"): trn, val, te = trn[:400], val[:100], te[:300]     # quick end-to-end check only
print("=== %s | %s | split %s seed %d lr %g | train %d val %d test %d | %s ===" % (
    TAG, MODEL, SPLIT, SEED, LR, len(trn), len(val), len(te), torch.cuda.get_device_name(0)), flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)
def mk(ix):
    enc = tok(list(X[ix]), truncation=True, max_length=128)
    return [{**{k: enc[k][i] for k in enc}, "labels": int(y[ix][i])} for i in range(len(ix))]
dtr, dva, dte = mk(trn), mk(val), mk(te)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
ck = os.path.join(W, "ckpt", TAG)
args = TrainingArguments(
    output_dir=ck, num_train_epochs=3, per_device_train_batch_size=16, per_device_eval_batch_size=128,
    learning_rate=LR, weight_decay=0.01, warmup_ratio=0.1, lr_scheduler_type="linear",
    seed=SEED, data_seed=SEED, eval_strategy="epoch", save_strategy="epoch",
    load_best_model_at_end=True, metric_for_best_model="f1", greater_is_better=True,
    save_total_limit=1, logging_steps=500, fp16=True, report_to=[], disable_tqdm=True)
metric = lambda p: {"f1": f1_score(p.label_ids, p.predictions.argmax(-1), average="weighted")}
trainer = Trainer(model=model, args=args, train_dataset=dtr, eval_dataset=dva,
                  data_collator=DataCollatorWithPadding(tok), compute_metrics=metric)
t0 = time.time(); trainer.train(); train_s = time.time() - t0
val_f1 = trainer.evaluate(dva)["eval_f1"]
logits = trainer.predict(dte).predictions
prob = torch.softmax(torch.tensor(logits, dtype=torch.float32), -1)[:, 1].numpy(); pred = logits.argmax(-1)
yt = y[te]; two = len(np.unique(yt)) == 2
res = {"tag": TAG, "split": SPLIT, "seed": SEED, "lr": LR, "model": MODEL, "device": torch.cuda.get_device_name(0),
       "n_train": int(len(trn)), "n_val": int(len(val)), "n_test": int(len(te)), "val_f1_weighted_best": float(val_f1),
       "accuracy": float(accuracy_score(yt, pred)), "f1_weighted": float(f1_score(yt, pred, average="weighted")),
       "f1_macro": float(f1_score(yt, pred, average="macro")) if two else None,
       "balanced_accuracy": float(balanced_accuracy_score(yt, pred)) if two else None,
       "roc_auc": float(roc_auc_score(yt, prob)) if two else None,
       "recall_factual": float(recall_score(yt, pred, pos_label=0)) if (yt == 0).any() else None,
       "recall_hoax": float(recall_score(yt, pred, pos_label=1)) if (yt == 1).any() else None,
       "train_time_s": train_s, "versions": {"torch": torch.__version__, "transformers": __import__("transformers").__version__}}
np.savez(os.path.join(OUTD, "out_pod_%s.npz" % TAG), pred=pred, prob=prob, true=yt, test_idx=te)
json.dump(res, open(RESULT, "w"), indent=2)
shutil.rmtree(ck, ignore_errors=True)
print("### %s: acc %.4f wF1 %.4f val %.4f | %.0fs" % (TAG, res["accuracy"], res["f1_weighted"], val_f1, train_s), flush=True)
os._exit(0)
