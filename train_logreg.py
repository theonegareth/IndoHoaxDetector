"""
Train Logistic Regression with TF-IDF for IndoHoaxDetector comparison.

This script trains a Logistic Regression classifier on the same preprocessed data and TF-IDF features,
with configurable regularization C parameter.

Usage:
    python train_logreg.py [--c C_VALUE] [--output OUTPUT_PATH]

Outputs:
    - logreg_model.pkl (or custom name): Trained Logistic Regression model
    - Evaluation metrics printed to console
"""

import sys
import os
import argparse
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split

# =========================
# CONFIG
# =========================

DEFAULT_MODEL_PATH = "logreg_model.pkl"
DEFAULT_VECTORIZER_PATH = "tfidf_vectorizer.pkl"
DEFAULT_TEXT_COL = "text_clean"
DEFAULT_LABEL_COL = "label_encoded"
DEFAULT_C = 1.0
INT_TO_STRING_LABEL = {0: "FAKTA", 1: "HOAX"}

# =========================
# LOADING UTILS
# =========================

def load_vectorizer(vectorizer_path: str):
    if not os.path.exists(vectorizer_path):
        print(f"[ERROR] Vectorizer file not found: {vectorizer_path}", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] Loading vectorizer from: {vectorizer_path}")
    vectorizer = joblib.load(vectorizer_path)
    return vectorizer

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

    df = df.rename(columns={text_col: "text", label_col: "true_label"})

    before = len(df)
    df = df[df["true_label"].isin([0, 1])]
    after = len(df)
    if after == 0:
        print("[ERROR] No rows with valid labels (0/1) after filtering.", file=sys.stderr)
        return pd.DataFrame()
    if after < before:
        print(f"[INFO] Filtered out {before - after} rows with invalid label values.")

    return df.reset_index(drop=True)

# =========================
# TRAINING LOGIC
# =========================

def train_logreg_on_labeled(
    df: pd.DataFrame,
    vectorizer,
    model_path: str,
    C: float = 1.0,
    random_state: int = 42,
    test_size: float = 0.2,
):
    print(f"[INFO] Using pre-cleaned text from dataset (no extra preprocessing).")
    print(f"[INFO] Logistic Regression C={C}")

    # Vectorize using existing TF-IDF
    print("[INFO] Vectorizing texts with existing TF-IDF...")
    X = vectorizer.transform(df["text"])
    y = df["true_label"]

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[INFO] Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # Train Logistic Regression
    print("[INFO] Training Logistic Regression...")
    start_time = time.time()
    model = LogisticRegression(C=C, max_iter=1000, random_state=random_state)
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"[INFO] Training completed in {training_time:.2f} seconds.")

    # Save model
    print(f"[INFO] Saving Logistic Regression model to: {model_path}")
    joblib.dump(model, model_path)

    # Evaluate on test set
    print("[INFO] Evaluating on test set...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # probability of class 1 (HOAX)

    # Compute metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Test Accuracy:  {acc:.4f}")
    print(f"Test Precision: {prec:.4f}")
    print(f"Test Recall:    {rec:.4f}")
    print(f"Test F1-score:  {f1:.4f}")

    print("\nTest Classification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["FAKTA(0)", "HOAX(1)"],
            digits=4,
        )
    )

    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_test, y_pred))

    # Return metrics for logging
    metrics = {
        "C": C,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "training_time": training_time,
        "train_size": X_train.shape[0],
        "test_size": X_test.shape[0],
    }
    return model, metrics

# =========================
# MAIN
# =========================

def main():
    parser = argparse.ArgumentParser(description="Train Logistic Regression with TF-IDF")
    parser.add_argument("--c", type=float, default=DEFAULT_C,
                        help="Regularization strength C (default: 1.0)")
    parser.add_argument("--output", type=str, default=DEFAULT_MODEL_PATH,
                        help="Path to save the trained model (default: logreg_model.pkl)")
    parser.add_argument("--vectorizer", type=str, default=DEFAULT_VECTORIZER_PATH,
                        help="Path to TF-IDF vectorizer (default: tfidf_vectorizer.pkl)")
    parser.add_argument("--data", type=str,
                        default="g:/My Drive/University Files/5th Semester/Data Science/Project/preprocessed_data_FINAL_FINAL.csv",
                        help="Path to preprocessed CSV (default: preprocessed_data_FINAL_FINAL.csv)")
    parser.add_argument("--text-col", type=str, default=DEFAULT_TEXT_COL,
                        help="Name of text column (default: text_clean)")
    parser.add_argument("--label-col", type=str, default=DEFAULT_LABEL_COL,
                        help="Name of label column (default: label_encoded)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Test split proportion (default: 0.2)")
    args = parser.parse_args()

    # Load vectorizer (reuse from Logistic Regression training)
    vectorizer = load_vectorizer(args.vectorizer)

    # Load data
    df = load_labeled_data(
        csv_path=args.data,
        text_col=args.text_col,
        label_col=args.label_col,
    )
    if df.empty:
        sys.exit(1)

    # Train Logistic Regression
    model, metrics = train_logreg_on_labeled(
        df=df,
        vectorizer=vectorizer,
        model_path=args.output,
        C=args.c,
        random_state=args.seed,
        test_size=args.test_size,
    )

    print(f"\n[INFO] Logistic Regression training complete. Model saved to {args.output}")
    print(f"[INFO] Metrics: {metrics}")

if __name__ == "__main__":
    main()