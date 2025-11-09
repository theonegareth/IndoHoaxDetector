"""
Helper usage from Jupyter:

Instead of `%run evaluate_model.py` (which passes Jupyter's --f arg and breaks argparse),
import this file and call:

    from evaluate_model import run_evaluation
    run_evaluation()

You can also override paths:

    run_evaluation(
        data_path="g:/My Drive/University Files/5th Semester/Data Science/Project/preprocessed_data_FINAL_FINAL.csv",
        text_col="text_clean",
        label_col="label_encoded",
        model_path="logreg_model.pkl",
        vectorizer_path="tfidf_vectorizer.pkl",
        max_show=5,
    )

This avoids the ipykernel_launcher.py --f error entirely.
"""

import argparse
import sys
import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

# =========================
# 1. CONFIG
# =========================

# Default paths (override via CLI args)
DEFAULT_MODEL_PATH = "logreg_model.pkl"
DEFAULT_VECTORIZER_PATH = "tfidf_vectorizer.pkl"

# Expected labeled CSV columns:
# For your case (preprocessed_data_FINAL_FINAL.csv):
# - text_col: "text_clean"
# - label_col: "label_encoded"  (0 = FAKTA, 1 = HOAX)
DEFAULT_TEXT_COL = "text_clean"
DEFAULT_LABEL_COL = "label_encoded"

INT_TO_STRING_LABEL = {0: "FAKTA", 1: "HOAX"}


# =========================
# 2. LOADING UTILS
# =========================

def load_model_and_vectorizer(
    model_path: str,
    vectorizer_path: str,
):
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(vectorizer_path):
        print(f"[ERROR] Vectorizer file not found: {vectorizer_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Loading model from: {model_path}")
    model = joblib.load(model_path)
    print(f"[INFO] Loading vectorizer from: {vectorizer_path}")
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


def load_labeled_data(
    csv_path: str,
    text_col: str,
    label_col: str,
) -> pd.DataFrame:
    """
    Load labeled evaluation data.

    Assumptions for your project:
    - text_col = 'text_clean' (already preprocessed exactly as during training)
    - label_col = 'label_encoded' (0 = FAKTA, 1 = HOAX)
    """
    if not os.path.exists(csv_path):
        print(f"[ERROR] Labeled CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Loading labeled data from: {csv_path}")
    df = pd.read_csv(csv_path)

    if text_col not in df.columns:
        print(f"[ERROR] Text column '{text_col}' not found. Available: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    if label_col not in df.columns:
        print(f"[ERROR] Label column '{label_col}' not found. Available: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    df = df[[text_col, label_col]].dropna()
    if df.empty:
        print("[ERROR] No valid rows after dropping NA in text/label.", file=sys.stderr)
        sys.exit(1)

    # For your file, label_encoded is already 0/1, so we keep as is
    df = df.rename(columns={text_col: "text", label_col: "true_label"})

    # Keep only rows with labels 0 or 1
    before = len(df)
    df = df[df["true_label"].isin([0, 1])]
    after = len(df)
    if after == 0:
        print("[ERROR] No rows with valid labels (0/1) after filtering.", file=sys.stderr)
        sys.exit(1)
    if after < before:
        print(f"[INFO] Filtered out {before - after} rows with invalid label values.", flush=True)

    return df.reset_index(drop=True)


# =========================
# 3. EVALUATION LOGIC
# =========================

def evaluate_model_on_labeled(
    df: pd.DataFrame,
    model,
    vectorizer,
    max_examples_to_show: int = 5,
):
    """
    Evaluate model on already-preprocessed texts.

    IMPORTANT:
    - We DO NOT re-clean or restem here because `text_clean` in your dataset
      is assumed to already match what the vectorizer was trained on.
    """
    print("[INFO] Using pre-cleaned text from dataset (no extra preprocessing).")

    # Vectorize using loaded TF-IDF (MUST use transform, not fit_transform)
    print("[INFO] Vectorizing texts with existing TF-IDF...")
    X = vectorizer.transform(df["text"])

    # Predictions
    print("[INFO] Running predictions...")
    probs = model.predict_proba(X)
    preds = model.predict(X)
    confidences = probs.max(axis=1)

    df["pred_label"] = preds
    df["pred_str"] = df["pred_label"].map(INT_TO_STRING_LABEL)
    df["true_str"] = df["true_label"].map(INT_TO_STRING_LABEL)
    df["confidence"] = confidences

    # --- Metrics ---
    print("\n===== CORE METRICS =====")
    acc = accuracy_score(df["true_label"], df["pred_label"])
    print(f"Accuracy: {acc:.4f}")

    print("\nClassification report (macro/micro F1, per-class metrics):")
    print(
        classification_report(
            df["true_label"],
            df["pred_label"],
            target_names=["FAKTA(0)", "HOAX(1)"],
            digits=4,
        )
    )

    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(df["true_label"], df["pred_label"]))

    # --- Error buckets ---
    fp = df[(df["true_label"] == 0) & (df["pred_label"] == 1)]
    fn = df[(df["true_label"] == 1) & (df["pred_label"] == 0)]

    print(f"\nTotal examples: {len(df)}")
    print(f"False Positives (FAKTA→HOAX): {len(fp)}")
    print(f"False Negatives (HOAX→FAKTA): {len(fn)}")

    # Show high-confidence mistakes for qualitative analysis
    def show_examples(sub_df, title: str):
        if sub_df.empty:
            print(f"\nNo {title} examples.")
            return
        print(f"\n===== {title} (up to {max_examples_to_show}) =====")
        sub_df_sorted = sub_df.sort_values("confidence", ascending=False).head(max_examples_to_show)
        for _, row in sub_df_sorted.iterrows():
            snippet = str(row["text"]).replace("\n", " ")
            if len(snippet) > 200:
                snippet = snippet[:200] + "..."
            print(
                f"- true={row['true_str']}, pred={row['pred_str']}, "
                f"conf={row['confidence']:.3f} :: {snippet}"
            )

    show_examples(fp, "High-confidence False Positives")
    show_examples(fn, "High-confidence False Negatives")

    print("\n[INFO] Evaluation complete.")
    return df


# =========================
# 4. CLI
# =========================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate IndoHoaxDetector (TF-IDF + Logistic Regression) on a labeled CSV "
            "and print accuracy, F1, confusion matrix, and sample errors.\n"
            "Defaults are set for preprocessed_data_FINAL_FINAL_FINAL.csv "
            "(text_clean, label_encoded).\n"
            "If --data is not provided, it defaults to 'g:/My Drive/University Files/5th Semester/Data Science/Project/preprocessed_data_FINAL_FINAL_FINAL.csv'."
        )
    )
    parser.add_argument(
        "--data",
        type=str,
        default="g:/My Drive/University Files/5th Semester/Data Science/Project/preprocessed_data_FINAL_FINAL_FINAL.csv",
        help=(
            "Path to labeled CSV containing preprocessed text and ground-truth labels. "
            "Default: g:/My Drive/University Files/5th Semester/Data Science/Project/preprocessed_data_FINAL_FINAL_FINAL.csv"
        ),
    )
    parser.add_argument(
        "--text-col",
        type=str,
        default=DEFAULT_TEXT_COL,
        help=f"Name of text column in CSV (default: {DEFAULT_TEXT_COL}).",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default=DEFAULT_LABEL_COL,
        help=f"Name of label column in CSV (default: {DEFAULT_LABEL_COL}).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to saved model .pkl (default: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument(
        "--vectorizer-path",
        type=str,
        default=DEFAULT_VECTORIZER_PATH,
        help=f"Path to saved TF-IDF vectorizer .pkl (default: {DEFAULT_VECTORIZER_PATH}).",
    )
    parser.add_argument(
        "--max-show",
        type=int,
        default=5,
        help="Max number of FP/FN examples to print for qualitative error analysis.",
    )
    return parser.parse_args()


# =========================
# 5. ENTRYPOINTS
# =========================

def main():
    """
    CLI entrypoint (for running from a real terminal).
    """
    args = parse_args()

    model, vectorizer = load_model_and_vectorizer(
        args.model_path,
        args.vectorizer_path,
    )
    df = load_labeled_data(
        csv_path=args.data,
        text_col=args.text_col,
        label_col=args.label_col,
    )

    evaluate_model_on_labeled(
        df=df,
        model=model,
        vectorizer=vectorizer,
        max_examples_to_show=args.max_show,
    )


def run_evaluation(
    data_path: str = "g:/My Drive/University Files/5th Semester/Data Science/Project/preprocessed_data_FINAL_FINAL_FINAL.csv",
    text_col: str = "text_clean",
    label_col: str = "label_encoded",
    model_path: str = "logreg_model.pkl",
    vectorizer_path: str = "tfidf_vectorizer.pkl",
    max_show: int = 5,
):
    """
    JUPYTER-FRIENDLY ENTRYPOINT.

    Call this from a notebook to avoid argparse / ipykernel --f issues.

    Example in a notebook cell (with notebook in the same folder as this file):

        from evaluate_model import run_evaluation
        run_evaluation()
    """
    model, vectorizer = load_model_and_vectorizer(
        model_path,
        vectorizer_path,
    )
    df = load_labeled_data(
        csv_path=data_path,
        text_col=text_col,
        label_col=label_col,
    )

    return evaluate_model_on_labeled(
        df=df,
        model=model,
        vectorizer=vectorizer,
        max_examples_to_show=max_show,
    )


if __name__ == "__main__":
    main()
