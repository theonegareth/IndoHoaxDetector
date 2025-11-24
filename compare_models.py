"""
Compare all trained models for IndoHoaxDetector.

Loads Logistic Regression, SVM, Random Forest, Naive Bayes, and IndoBERT (if available),
evaluates them on the same test set, and prints a comparison table.

Usage:
    python compare_models.py

Outputs:
    - Comparison table of metrics
    - Detailed reports for each model
"""

import sys
import os
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.model_selection import train_test_split

# =========================
# CONFIG
# =========================

DEFAULT_VECTORIZER_PATH = "tfidf_vectorizer.pkl"
DEFAULT_TEXT_COL = "text_clean"
DEFAULT_LABEL_COL = "label_encoded"

MODEL_PATHS = {
    "Logistic Regression": "logreg_model.pkl",
    "Linear SVM": "svm_model.pkl",
    "Random Forest": "rf_model.pkl",
    "Multinomial Naive Bayes": "nb_model.pkl",
    "IndoBERT": "indobert_model",  # Directory for transformers
}

# =========================
# LOADING UTILS
# =========================

def load_vectorizer(vectorizer_path: str):
    if not os.path.exists(vectorizer_path):
        print(f"[ERROR] Vectorizer file not found: {vectorizer_path}", file=sys.stderr)
        return None
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

def load_sklearn_model(model_path: str):
    if not os.path.exists(model_path):
        print(f"[WARNING] Model file not found: {model_path}")
        return None
    print(f"[INFO] Loading model from: {model_path}")
    model = joblib.load(model_path)
    return model

def load_indobert_model(model_path: str):
    if not os.path.exists(model_path):
        print(f"[WARNING] IndoBERT model directory not found: {model_path}")
        return None, None
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        print(f"[INFO] Loading IndoBERT from: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        return model, tokenizer
    except ImportError:
        print("[WARNING] Transformers not installed. Skipping IndoBERT.")
        return None, None

# =========================
# EVALUATION LOGIC
# =========================

def evaluate_model(
    model,
    X_test,
    y_test,
    model_name: str,
    is_indobert: bool = False,
    tokenizer = None,
):
    if is_indobert:
        # For IndoBERT, need to tokenize
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        # Tokenize test data
        test_encodings = tokenizer(
            X_test.tolist(),
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**test_encodings)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
    else:
        preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='macro')
    rec = recall_score(y_test, preds, average='macro')
    f1 = f1_score(y_test, preds, average='macro')

    print(f"\n=== {model_name} ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (macro): {prec:.4f}")
    print(f"Recall (macro): {rec:.4f}")
    print(f"F1 (macro): {f1:.4f}")

    return {
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
    }

# =========================
# MAIN
# =========================

def main():
    # Paths
    data_path = "g:/My Drive/University Files/5th Semester/Data Science/Project/preprocessed_data_FINAL_FINAL.csv"
    text_col = DEFAULT_TEXT_COL
    label_col = DEFAULT_LABEL_COL
    vectorizer_path = DEFAULT_VECTORIZER_PATH

    # Load data
    df = load_labeled_data(
        csv_path=data_path,
        text_col=text_col,
        label_col=label_col,
    )
    if df.empty:
        sys.exit(1)

    # Split into test set (20% for fair comparison)
    _, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["true_label"])
    print(f"[INFO] Using {len(test_df)} samples for testing.")

    X_test_text = test_df["text"]
    y_test = test_df["true_label"]

    # Load vectorizer for TF-IDF models
    vectorizer = load_vectorizer(vectorizer_path)
    if vectorizer:
        X_test_tfidf = vectorizer.transform(X_test_text)

    results = []

    # Evaluate each model
    for model_name, model_path in MODEL_PATHS.items():
        if model_name == "IndoBERT":
            model, tokenizer = load_indobert_model(model_path)
            if model:
                result = evaluate_model(model, X_test_text, y_test, model_name, is_indobert=True, tokenizer=tokenizer)
                results.append(result)
        else:
            model = load_sklearn_model(model_path)
            if model and vectorizer:
                result = evaluate_model(model, X_test_tfidf, y_test, model_name)
                results.append(result)

    # Print comparison table
    if results:
        results_df = pd.DataFrame(results)
        print("\n" + "="*50)
        print("MODEL COMPARISON SUMMARY")
        print("="*50)
        print(results_df.to_string(index=False))
    else:
        print("[ERROR] No models could be loaded.")

if __name__ == "__main__":
    main()