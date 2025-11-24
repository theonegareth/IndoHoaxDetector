"""
Train Random Forest with TF-IDF for IndoHoaxDetector comparison.

This script trains a Random Forest classifier on the same preprocessed data and TF-IDF features
as the Logistic Regression and SVM models, for performance comparison.

Usage:
    python train_rf.py

Outputs:
    - rf_model.pkl: Trained Random Forest model
    - Evaluation metrics printed to console
"""

import sys
import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

# =========================
# CONFIG (same as Logistic Regression)
# =========================

DEFAULT_MODEL_PATH = "rf_model.pkl"
DEFAULT_VECTORIZER_PATH = "tfidf_vectorizer.pkl"
DEFAULT_TEXT_COL = "text_clean"
DEFAULT_LABEL_COL = "label_encoded"

INT_TO_STRING_LABEL = {0: "FAKTA", 1: "HOAX"}

# =========================
# LOADING UTILS (reuse from evaluate_model.py)
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

def train_rf_on_labeled(
    df: pd.DataFrame,
    vectorizer,
    model_path: str,
):
    print("[INFO] Using pre-cleaned text from dataset (no extra preprocessing).")

    # Vectorize using existing TF-IDF
    print("[INFO] Vectorizing texts with existing TF-IDF...")
    X = vectorizer.transform(df["text"])
    y = df["true_label"]

    # Train Random Forest
    print("[INFO] Training Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)  # Use parallel processing
    model.fit(X, y)

    # Save model
    print(f"[INFO] Saving Random Forest model to: {model_path}")
    joblib.dump(model, model_path)

    # Quick evaluation on training data (for sanity check)
    print("[INFO] Quick evaluation on training data:")
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    print(f"Training Accuracy: {acc:.4f}")

    print("\nTraining Classification report:")
    print(
        classification_report(
            y,
            preds,
            target_names=["FAKTA(0)", "HOAX(1)"],
            digits=4,
        )
    )

    return model

# =========================
# MAIN
# =========================

def main():
    # Paths (same as Logistic Regression)
    data_path = "g:/My Drive/University Files/5th Semester/Data Science/Project/preprocessed_data_FINAL_FINAL.csv"
    text_col = DEFAULT_TEXT_COL
    label_col = DEFAULT_LABEL_COL
    vectorizer_path = DEFAULT_VECTORIZER_PATH
    model_path = DEFAULT_MODEL_PATH

    # Load vectorizer (reuse from Logistic Regression training)
    vectorizer = load_vectorizer(vectorizer_path)

    # Load data
    df = load_labeled_data(
        csv_path=data_path,
        text_col=text_col,
        label_col=label_col,
    )
    if df.empty:
        sys.exit(1)

    # Train Random Forest
    model = train_rf_on_labeled(
        df=df,
        vectorizer=vectorizer,
        model_path=model_path,
    )

    print(f"\n[INFO] Random Forest training complete. Model saved to {model_path}")

if __name__ == "__main__":
    main()