"""
Train IndoBERT fine-tuned for IndoHoaxDetector comparison.

This script fine-tunes IndoBERT (Indonesian BERT) on the same preprocessed data
for binary classification (HOAX/FAKTA), for performance comparison.

Note: Requires GPU for efficient training. If GPU not available, training will be slow on CPU.
Install: pip install transformers torch datasets

Usage:
    python train_indobert.py

Outputs:
    - indobert_model/: Saved fine-tuned IndoBERT model
    - Evaluation metrics printed to console
"""

import sys
import os
from typing import Tuple

import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import Dataset
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

# =========================
# CONFIG
# =========================

DEFAULT_MODEL_PATH = "indobert_model"
DEFAULT_TEXT_COL = "text_clean"
DEFAULT_LABEL_COL = "label_encoded"

# IndoBERT model (Indonesian BERT)
MODEL_NAME = "indobenchmark/indobert-base-p1"  # Or "indolem/indobert-base-uncased" if preferred

# Training params (adjust for GPU/CPU)
MAX_LENGTH = 128
BATCH_SIZE = 16  # Reduce if OOM
EPOCHS = 3
LEARNING_RATE = 2e-5

# =========================
# LOADING UTILS
# =========================

def load_labeled_data(
    csv_path: str,
    text_col: str,
    label_col: str,
) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        print(f"[ERROR] Labeled CSV not found: {csv_path}", file=sys.stderr)
        return pd.DataFrame()

    print(f"[INFO] Loading labeled data from: {csv_path}")
    df = pd.read_csv(csv_path)

    if text_col not in df.columns:
        print(f"[ERROR] Text column '{text_col}' not found. Available: {list(df.columns)}", file=sys.stderr)
        return pd.DataFrame()

    if label_col not in df.columns:
        print(f"[ERROR] Label column '{label_col}' not found. Available: {list(df.columns)}", file=sys.stderr)
        return pd.DataFrame()

    df = df[[text_col, label_col]].dropna()
    if df.empty:
        print("[ERROR] No valid rows after dropping NA in text/label.", file=sys.stderr)
        return pd.DataFrame()

    df = df.rename(columns={text_col: "text", label_col: "label"})

    before = len(df)
    df = df[df["label"].isin([0, 1])]
    after = len(df)
    if after == 0:
        print("[ERROR] No rows with valid labels (0/1) after filtering.", file=sys.stderr)
        return pd.DataFrame()
    if after < before:
        print(f"[INFO] Filtered out {before - after} rows with invalid label values.")

    return df.reset_index(drop=True)

# =========================
# TOKENIZATION
# =========================

def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

# =========================
# TRAINING LOGIC
# =========================

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

def train_indobert_on_labeled(
    df: pd.DataFrame,
    model_path: str,
):
    print("[INFO] Using pre-cleaned text from dataset.")

    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    if device.type == "cpu":
        print("[WARNING] GPU not available. Training will be slow on CPU.")

    # Load tokenizer and model
    print(f"[INFO] Loading IndoBERT: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)

    # Prepare dataset
    dataset = Dataset.from_pandas(df)
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"],
    )
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

    # Split into train/val (simple 80/20)
    split_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training args
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        logging_dir="./logs",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train
    print("[INFO] Starting fine-tuning...")
    trainer.train()

    # Save model
    print(f"[INFO] Saving fine-tuned IndoBERT to: {model_path}")
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)

    # Evaluate on val set
    print("[INFO] Evaluating on validation set...")
    predictions = trainer.predict(val_dataset)
    preds = predictions.predictions.argmax(-1)
    labels = predictions.label_ids

    print("Validation Classification report:")
    print(
        classification_report(
            labels,
            preds,
            target_names=["FAKTA(0)", "HOAX(1)"],
            digits=4,
        )
    )

    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(labels, preds))

    return model, tokenizer

# =========================
# MAIN
# =========================

def main():
    # Paths
    data_path = "g:/My Drive/University Files/5th Semester/Data Science/Project/preprocessed_data_FINAL_FINAL.csv"
    text_col = DEFAULT_TEXT_COL
    label_col = DEFAULT_LABEL_COL
    model_path = DEFAULT_MODEL_PATH

    # Load data
    df = load_labeled_data(
        csv_path=data_path,
        text_col=text_col,
        label_col=label_col,
    )
    if df.empty:
        sys.exit(1)

    # Train IndoBERT
    model, tokenizer = train_indobert_on_labeled(
        df=df,
        model_path=model_path,
    )

    print(f"\n[INFO] IndoBERT fine-tuning complete. Model saved to {model_path}")

if __name__ == "__main__":
    main()