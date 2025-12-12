"""
Train sklearn models for IndoHoaxDetector comparison.

This script trains Logistic Regression, SVM, Random Forest, and Naive Bayes
on the same preprocessed data for binary classification (HOAX/FAKTA).

Usage:
    python train_sklearn.py

Outputs:
    - tfidf_vectorizer.pkl: TF-IDF vectorizer
    - logreg_model.pkl: Logistic Regression model
    - svm_model.pkl: Linear SVM model
    - rf_model.pkl: Random Forest model
    - nb_model.pkl: Multinomial Naive Bayes model
"""

import sys
import os
from typing import Tuple

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# =========================
# CONFIG
# =========================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(SCRIPT_DIR, "preprocessed_data_FINAL_FINAL.csv")
DEFAULT_TEXT_COL = "text_clean"
DEFAULT_LABEL_COL = "label_encoded"

# TF-IDF params
MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)

# Model params
RANDOM_STATE = 42

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
# TRAINING LOGIC
# =========================

def train_sklearn_models(df: pd.DataFrame):
    print("[INFO] Preparing data for sklearn models.")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=RANDOM_STATE, stratify=df["label"]
    )
    print(f"[INFO] Train size: {len(X_train)}, Test size: {len(X_test)}")

    # TF-IDF vectorization
    print("[INFO] Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Save vectorizer
    vectorizer_path = os.path.join(SCRIPT_DIR, "tfidf_vectorizer.pkl")
    joblib.dump(vectorizer, vectorizer_path)
    print(f"[INFO] Saved TF-IDF vectorizer to: {vectorizer_path}")

    models = {
        "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        "Linear SVM": LinearSVC(random_state=RANDOM_STATE, max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100),
        "Multinomial Naive Bayes": MultinomialNB(),
    }

    for name, model in models.items():
        print(f"[INFO] Training {name}...")
        model.fit(X_train_tfidf, y_train)

        # Quick eval on test set
        preds = model.predict(X_test_tfidf)
        print(f"[INFO] {name} test accuracy: {model.score(X_test_tfidf, y_test):.4f}")

        # Save model
        if name == "Logistic Regression":
            model_path = os.path.join(SCRIPT_DIR, "logreg_model.pkl")
        elif name == "Linear SVM":
            model_path = os.path.join(SCRIPT_DIR, "svm_model.pkl")
        elif name == "Random Forest":
            model_path = os.path.join(SCRIPT_DIR, "rf_model.pkl")
        elif name == "Multinomial Naive Bayes":
            model_path = os.path.join(SCRIPT_DIR, "nb_model.pkl")
        else:
            model_path = os.path.join(SCRIPT_DIR, f"{name.lower().replace(' ', '_')}_model.pkl")
        joblib.dump(model, model_path)
        print(f"[INFO] Saved {name} to: {model_path}")

# =========================
# MAIN
# =========================

def main():
    # Paths
    data_path = DEFAULT_DATA_PATH
    text_col = DEFAULT_TEXT_COL
    label_col = DEFAULT_LABEL_COL

    # Load data
    df = load_labeled_data(
        csv_path=data_path,
        text_col=text_col,
        label_col=label_col,
    )
    if df.empty:
        sys.exit(1)

    # Train models
    train_sklearn_models(df)

    print(f"\n[INFO] Sklearn models training complete.")

if __name__ == "__main__":
    main()