"""
Run experiments with Logistic Regression across different C values.

This script loops over a list of C values, trains a Logistic Regression model for each,
evaluates on test set, and records metrics (accuracy, precision, recall, F1, training time)
into a CSV file. It also optionally generates a plot of accuracy vs C.

Usage:
    python run_logreg_experiments.py [--c-values 0.01 0.1 1 10] [--output results/logreg_c_sweep.csv]

Outputs:
    - CSV file with metrics per C value
    - Plot (optional) saved as results/logreg_c_sweep.png
"""

import sys
import os
import argparse
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Import the training utilities from train_logreg
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_logreg import load_vectorizer, load_labeled_data

# =========================
# CONFIG
# =========================

DEFAULT_C_VALUES = [0.01, 0.1, 1.0, 10.0]
DEFAULT_VECTORIZER_PATH = "tfidf_vectorizer.pkl"
DEFAULT_DATA_PATH = "g:/My Drive/University Files/5th Semester/Data Science/Project/preprocessed_data_FINAL_FINAL.csv"
DEFAULT_TEXT_COL = "text_clean"
DEFAULT_LABEL_COL = "label_encoded"
DEFAULT_OUTPUT_CSV = "results/logreg_c_sweep.csv"
DEFAULT_OUTPUT_PLOT = "results/logreg_c_sweep.png"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# =========================
# EXPERIMENT RUNNER
# =========================

def run_experiment(C, vectorizer, X_train, X_test, y_train, y_test):
    """Train Logistic Regression with given C and return metrics."""
    from sklearn.linear_model import LogisticRegression

    print(f"[INFO] Training Logistic Regression with C={C}...")
    start_time = time.time()
    model = LogisticRegression(C=C, max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    metrics = {
        "C": C,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "training_time_sec": training_time,
        "train_size": X_train.shape[0],
        "test_size": X_test.shape[0],
    }
    return metrics, model

def main():
    parser = argparse.ArgumentParser(description="Run Logistic Regression C sweep")
    parser.add_argument("--c-values", nargs="+", type=float, default=DEFAULT_C_VALUES,
                        help="List of C values to test (default: 0.01 0.1 1 10)")
    parser.add_argument("--output-csv", type=str, default=DEFAULT_OUTPUT_CSV,
                        help="Path to output CSV (default: results/logreg_c_sweep.csv)")
    parser.add_argument("--plot", action="store_true", default=True,
                        help="Generate plot of accuracy vs C (default: True)")
    parser.add_argument("--output-plot", type=str, default=DEFAULT_OUTPUT_PLOT,
                        help="Path to output plot (default: results/logreg_c_sweep.png)")
    parser.add_argument("--vectorizer", type=str, default=DEFAULT_VECTORIZER_PATH,
                        help="Path to TF-IDF vectorizer (default: tfidf_vectorizer.pkl)")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_PATH,
                        help="Path to preprocessed CSV (default: preprocessed_data_FINAL_FINAL.csv)")
    parser.add_argument("--text-col", type=str, default=DEFAULT_TEXT_COL,
                        help="Name of text column (default: text_clean)")
    parser.add_argument("--label-col", type=str, default=DEFAULT_LABEL_COL,
                        help="Name of label column (default: label_encoded)")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_csv) if os.path.dirname(args.output_csv) else ".", exist_ok=True)
    if args.plot:
        os.makedirs(os.path.dirname(args.output_plot) if os.path.dirname(args.output_plot) else ".", exist_ok=True)

    # Load vectorizer and data
    print("[INFO] Loading vectorizer and data...")
    vectorizer = load_vectorizer(args.vectorizer)
    df = load_labeled_data(
        csv_path=args.data,
        text_col=args.text_col,
        label_col=args.label_col,
    )
    if df.empty:
        sys.exit(1)

    # Vectorize
    X = vectorizer.transform(df["text"])
    y = df["true_label"]

    # Split once for all experiments (consistent across C values)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[INFO] Data split: train={X_train.shape[0]}, test={X_test.shape[0]}")

    # Run experiments
    results = []
    for C in args.c_values:
        print(f"\n{'='*60}")
        print(f"Experiment C = {C}")
        print('='*60)
        try:
            metrics, model = run_experiment(C, vectorizer, X_train, X_test, y_train, y_test)
            results.append(metrics)
            print(f"  Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, "
                  f"Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
            # Optionally save each model
            # joblib.dump(model, f"results/logreg_C{C}.pkl")
        except Exception as e:
            print(f"[ERROR] Failed for C={C}: {e}")
            continue

    if not results:
        print("[ERROR] No experiments succeeded.")
        sys.exit(1)

    # Create DataFrame and save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_csv, index=False)
    print(f"\n[INFO] Results saved to {args.output_csv}")
    print(results_df.to_string(index=False))

    # Plot if requested
    if args.plot:
        plt.figure(figsize=(8, 5))
        plt.plot(results_df["C"], results_df["accuracy"], marker='o', label="Accuracy")
        plt.plot(results_df["C"], results_df["precision"], marker='s', label="Precision")
        plt.plot(results_df["C"], results_df["recall"], marker='^', label="Recall")
        plt.plot(results_df["C"], results_df["f1"], marker='d', label="F1")
        plt.xscale('log')
        plt.xlabel("Regularization C (log scale)")
        plt.ylabel("Score")
        plt.title("Logistic Regression Performance vs Regularization C")
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.output_plot, dpi=300)
        print(f"[INFO] Plot saved to {args.output_plot}")
        plt.show()

    # Print summary
    best_row = results_df.loc[results_df['accuracy'].idxmax()]
    print(f"\n[SUMMARY] Best accuracy {best_row['accuracy']:.4f} at C={best_row['C']}")
    print("[DONE] Experiment sweep completed.")

if __name__ == "__main__":
    main()