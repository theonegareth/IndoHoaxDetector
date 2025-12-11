"""
Lightweight model comparison for IndoHoaxDetector.

Evaluates each model sequentially, unloading after each to save memory.
Optionally, you can specify which models to run via command line.

Usage:
    python compare_models_light.py [--models lr svm rf nb indobert] [--output results/comparison_light.csv]

If no --models argument is given, runs all models.
"""

import sys
import os
import gc
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import joblib

# =========================
# CONFIG
# =========================

DEFAULT_VECTORIZER_PATH = "tfidf_vectorizer.pkl"
DEFAULT_TEXT_COL = "text_clean"
DEFAULT_LABEL_COL = "label_encoded"

MODEL_PATHS = {
    "lr": ("Logistic Regression", "logreg_model.pkl"),
    "svm": ("Linear SVM", "svm_model.pkl"),
    "rf": ("Random Forest", "rf_model.pkl"),
    "nb": ("Multinomial Naive Bayes", "nb_model.pkl"),
    "indobert": ("IndoBERT", "indobert_model"),
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

def load_labeled_data(csv_path: str, text_col: str, label_col: str) -> pd.DataFrame:
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
    df = df.rename(columns={text_col: "text", label_col: "true_label"})
    df = df[df["true_label"].isin([0, 1])]
    if df.empty:
        print("[ERROR] No rows with valid labels (0/1) after filtering.", file=sys.stderr)
        return pd.DataFrame()

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

def evaluate_sklearn(model, X_test, y_test, model_name: str):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='macro')
    rec = recall_score(y_test, preds, average='macro')
    f1 = f1_score(y_test, preds, average='macro')
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Precision (macro): {prec:.4f}")
    print(f"  Recall (macro): {rec:.4f}")
    print(f"  F1 (macro): {f1:.4f}")
    return {"Model": model_name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}

def evaluate_indobert(model, tokenizer, X_test_text, y_test, model_name: str):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Process in batches to avoid memory overflow
    batch_size = 32
    preds = []
    for i in range(0, len(X_test_text), batch_size):
        batch_texts = X_test_text.iloc[i:i+batch_size].tolist()
        encodings = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = model(**encodings)
            batch_preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            preds.extend(batch_preds)
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    preds = np.array(preds)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='macro')
    rec = recall_score(y_test, preds, average='macro')
    f1 = f1_score(y_test, preds, average='macro')
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Precision (macro): {prec:.4f}")
    print(f"  Recall (macro): {rec:.4f}")
    print(f"  F1 (macro): {f1:.4f}")
    return {"Model": model_name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}

# =========================
# MAIN
# =========================

def main():
    parser = argparse.ArgumentParser(description="Lightweight model comparison")
    parser.add_argument("--models", nargs="+", choices=["lr", "svm", "rf", "nb", "indobert"],
                        default=["lr", "svm", "rf", "nb", "indobert"],
                        help="Models to evaluate (default: all)")
    parser.add_argument("--output", type=str, default="results/model_comparison_light.csv",
                        help="CSV file to save results")
    parser.add_argument("--data", type=str,
                        default="g:/My Drive/University Files/5th Semester/Data Science/Project/preprocessed_data_FINAL_FINAL.csv",
                        help="Path to preprocessed CSV")
    parser.add_argument("--text-col", type=str, default=DEFAULT_TEXT_COL,
                        help="Text column name")
    parser.add_argument("--label-col", type=str, default=DEFAULT_LABEL_COL,
                        help="Label column name")
    parser.add_argument("--vectorizer", type=str, default=DEFAULT_VECTORIZER_PATH,
                        help="Path to TF-IDF vectorizer")
    args = parser.parse_args()

    # Load data once
    df = load_labeled_data(args.data, args.text_col, args.label_col)
    if df.empty:
        sys.exit(1)

    # Split into test set (20% for fair comparison)
    _, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["true_label"])
    print(f"[INFO] Using {len(test_df)} samples for testing.")
    X_test_text = test_df["text"]
    y_test = test_df["true_label"]

    # Load vectorizer for TF-IDF models
    vectorizer = load_vectorizer(args.vectorizer)
    if vectorizer is None:
        print("[ERROR] Vectorizer not found. Exiting.")
        sys.exit(1)
    X_test_tfidf = vectorizer.transform(X_test_text)

    results = []

    # Evaluate each requested model
    for model_key in args.models:
        model_name, model_path = MODEL_PATHS[model_key]
        print(f"\n[INFO] Evaluating {model_name}...")

        if model_key == "indobert":
            model, tokenizer = load_indobert_model(model_path)
            if model is None:
                print(f"[WARNING] IndoBERT not available, skipping.")
                continue
            result = evaluate_indobert(model, tokenizer, X_test_text, y_test, model_name)
            # Clean up
            del model, tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            model = load_sklearn_model(model_path)
            if model is None:
                print(f"[WARNING] Model {model_name} not found, skipping.")
                continue
            result = evaluate_sklearn(model, X_test_tfidf, y_test, model_name)
            del model

        results.append(result)
        gc.collect()  # encourage garbage collection

    # Print summary
    if results:
        results_df = pd.DataFrame(results)
        print("\n" + "="*60)
        print("MODEL COMPARISON SUMMARY")
        print("="*60)
        print(results_df.to_string(index=False))

        # Save to CSV
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        results_df.to_csv(args.output, index=False)
        print(f"\n[INFO] Results saved to {args.output}")
    else:
        print("[ERROR] No models were evaluated.")

if __name__ == "__main__":
    main()