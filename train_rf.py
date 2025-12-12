#!/usr/bin/env python3
"""
Train Random Forest model for IndoHoaxDetector with configurable n_estimators.

This script trains a Random Forest model with a specified number of trees,
evaluates it using 5-fold cross-validation, and saves the model and metrics.

Usage:
    python train_rf.py --n_estimators 100
"""

import argparse
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# =========================
# CONFIGURATION
# =========================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(SCRIPT_DIR, "preprocessed_data_FINAL_FINAL.csv")
DEFAULT_TEXT_COL = "text_clean"
DEFAULT_LABEL_COL = "label_encoded"
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results")

# Cross-validation parameters
CV_FOLDS = 5
RANDOM_STATE = 42

# =========================
# DATA LOADING
# =========================

def load_data(
    csv_path: str,
    text_col: str,
    label_col: str
) -> pd.DataFrame:
    """Load and validate the dataset."""
    if not os.path.exists(csv_path):
        print(f"[ERROR] Data file not found: {csv_path}", file=sys.stderr)
        return pd.DataFrame()
    
    print(f"[INFO] Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Validate columns
    if text_col not in df.columns:
        print(f"[ERROR] Text column '{text_col}' not found. Available: {list(df.columns)}", file=sys.stderr)
        return pd.DataFrame()
    
    if label_col not in df.columns:
        print(f"[ERROR] Label column '{label_col}' not found. Available: {list(df.columns)}", file=sys.stderr)
        return pd.DataFrame()
    
    # Select and clean data
    df = df[[text_col, label_col]].dropna()
    if df.empty:
        print("[ERROR] No valid rows after dropping NA values.", file=sys.stderr)
        return pd.DataFrame()
    
    # Rename columns for consistency
    df = df.rename(columns={text_col: "text", label_col: "label"})
    
    # Validate labels (should be 0 and 1)
    before = len(df)
    df = df[df["label"].isin([0, 1])]
    after = len(df)
    
    if after == 0:
        print("[ERROR] No rows with valid labels (0/1) after filtering.", file=sys.stderr)
        return pd.DataFrame()
    
    if after < before:
        print(f"[INFO] Filtered out {before - after} rows with invalid label values.")
    
    print(f"[INFO] Loaded {len(df)} valid samples.")
    print(f"[INFO] Class distribution: {df['label'].value_counts().to_dict()}")
    
    return df.reset_index(drop=True)

# =========================
# MODEL TRAINING AND EVALUATION
# =========================

def train_and_evaluate(
    df: pd.DataFrame,
    n_estimators: int,
    max_features: int,
    ngram_range: tuple,
    output_dir: str
) -> Dict[str, Any]:
    """Train Random Forest model and evaluate with cross-validation."""
    
    # Prepare data
    X = df["text"]
    y = df["label"]
    
    print(f"[INFO] Vectorizing text data with TF-IDF...")
    print(f"[INFO] max_features={max_features}, ngram_range={ngram_range}")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range
    )
    X_tfidf = vectorizer.fit_transform(X)
    
    print(f"[INFO] Training Random Forest with n_estimators={n_estimators}")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    # Perform cross-validation manually for better control
    print(f"[INFO] Running {CV_FOLDS}-fold cross-validation...")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    start_time = time.time()
    
    fold = 1
    for train_idx, val_idx in cv.split(X_tfidf, y):
        print(f"[INFO] Training fold {fold}/{CV_FOLDS}...")
        
        # Split data
        X_train_fold = X_tfidf[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X_tfidf[val_idx]
        y_val_fold = y.iloc[val_idx]
        
        # Train model on this fold
        fold_model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        fold_model.fit(X_train_fold, y_train_fold)
        
        # Predict and calculate metrics
        y_pred = fold_model.predict(X_val_fold)
        
        accuracy_scores.append(accuracy_score(y_val_fold, y_pred))
        precision_scores.append(precision_score(y_val_fold, y_pred, average='binary', zero_division=0))
        recall_scores.append(recall_score(y_val_fold, y_pred, average='binary', zero_division=0))
        f1_scores.append(f1_score(y_val_fold, y_pred, average='binary', zero_division=0))
        
        fold += 1
    
    training_duration = time.time() - start_time
    
    # Calculate mean metrics
    metrics = {
        'n_estimators': n_estimators,
        'accuracy_mean': float(np.mean(accuracy_scores)),
        'accuracy_std': float(np.std(accuracy_scores)),
        'precision_mean': float(np.mean(precision_scores)),
        'precision_std': float(np.std(precision_scores)),
        'recall_mean': float(np.mean(recall_scores)),
        'recall_std': float(np.std(recall_scores)),
        'f1_mean': float(np.mean(f1_scores)),
        'f1_std': float(np.std(f1_scores)),
        'training_duration': training_duration,
        'cv_folds': CV_FOLDS,
        'n_samples': len(df),
        'timestamp': datetime.now().isoformat()
    }
    
    # Print results
    print("\n" + "="*60)
    print("CROSS-VALIDATION RESULTS")
    print("="*60)
    print(f"n_estimators: {n_estimators}")
    print(f"Accuracy:  {metrics['accuracy_mean']:.4f} ± {metrics['accuracy_std']:.4f}")
    print(f"Precision: {metrics['precision_mean']:.4f} ± {metrics['precision_std']:.4f}")
    print(f"Recall:    {metrics['recall_mean']:.4f} ± {metrics['recall_std']:.4f}")
    print(f"F1 Score:  {metrics['f1_mean']:.4f} ± {metrics['f1_std']:.4f}")
    print(f"Training time: {training_duration:.2f} seconds")
    print("="*60)
    
    # Save model and vectorizer
    print(f"[INFO] Saving model artifacts...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save TF-IDF vectorizer
    vectorizer_path = os.path.join(output_dir, f"tfidf_vectorizer_rf_n{n_estimators}_mf{max_features}_ng{ngram_range[0]}-{ngram_range[1]}.pkl")
    joblib.dump(vectorizer, vectorizer_path)
    
    # Train final model on full dataset
    print(f"[INFO] Training final model on full dataset...")
    final_model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    final_model.fit(X_tfidf, y)
    
    # Save final model
    model_path = os.path.join(output_dir, f"rf_model_n{n_estimators}_mf{max_features}_ng{ngram_range[0]}-{ngram_range[1]}.pkl")
    joblib.dump(final_model, model_path)
    
    print(f"[INFO] Saved vectorizer: {vectorizer_path}")
    print(f"[INFO] Saved model: {model_path}")
    
    return metrics

# =========================
# MAIN
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="Train Random Forest model for hoax detection with configurable n_estimators."
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        required=True,
        help="Number of trees in the Random Forest"
    )
    parser.add_argument(
        "--max_features",
        type=int,
        default=5000,
        help="Maximum number of features for TF-IDF vectorizer (default: 5000)"
    )
    parser.add_argument(
        "--ngram_min",
        type=int,
        default=1,
        help="Minimum n-gram size for TF-IDF (default: 1)"
    )
    parser.add_argument(
        "--ngram_max",
        type=int,
        default=2,
        help="Maximum n-gram size for TF-IDF (default: 2)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help=f"Path to CSV file (default: {DEFAULT_DATA_PATH})"
    )
    parser.add_argument(
        "--text_col",
        type=str,
        default=DEFAULT_TEXT_COL,
        help=f"Name of text column (default: {DEFAULT_TEXT_COL})"
    )
    parser.add_argument(
        "--label_col",
        type=str,
        default=DEFAULT_LABEL_COL,
        help=f"Name of label column (default: {DEFAULT_LABEL_COL})"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for models and results (default: {DEFAULT_OUTPUT_DIR})"
    )
    
    args = parser.parse_args()
    
    # Validate n_estimators value
    if args.n_estimators <= 0:
        print(f"[ERROR] n_estimators must be positive. Got: {args.n_estimators}", file=sys.stderr)
        sys.exit(1)
    
    print(f"[INFO] Starting training with n_estimators={args.n_estimators}")
    print(f"[INFO] TF-IDF parameters: max_features={args.max_features}, ngram_range=({args.ngram_min},{args.ngram_max})")
    print(f"[INFO] Data path: {args.data_path}")
    print(f"[INFO] Output directory: {args.output_dir}")
    
    # Load data
    df = load_data(args.data_path, args.text_col, args.label_col)
    if df.empty:
        sys.exit(1)
    
    # Train and evaluate
    try:
        ngram_range = (args.ngram_min, args.ngram_max)
        metrics = train_and_evaluate(
            df,
            args.n_estimators,
            args.max_features,
            ngram_range,
            args.output_dir
        )
        
        # Save metrics to JSON for easy parsing
        import json
        metrics_path = os.path.join(args.output_dir, f"rf_metrics_n{args.n_estimators}_mf{args.max_features}_ng{args.ngram_min}-{args.ngram_max}.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"[INFO] Saved metrics to: {metrics_path}")
        print("[INFO] Training completed successfully!")
        
    except Exception as e:
        print(f"[ERROR] Training failed: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()