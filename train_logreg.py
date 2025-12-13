#!/usr/bin/env python3
"""Recreate the Logistic Regression training workflow for IndoHoaxDetector."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(SCRIPT_DIR, "preprocessed_data_FINAL_FINAL.csv")
DEFAULT_TEXT_COL = "text_clean"
DEFAULT_LABEL_COL = "label_encoded"
DEFAULT_MAX_FEATURES = 5000
DEFAULT_NGRAM_RANGE: Tuple[int, int] = (1, 2)
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_CV_FOLDS = 5
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results", "train_logreg")


def sanitize_c_value(c_value: float) -> str:
    return f"{c_value:.3g}".replace("-", "neg").replace(".", "_")


def ensure_directory(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def clean_dataframe(df: pd.DataFrame, text_col: str, label_col: str) -> pd.DataFrame:
    if text_col not in df.columns or label_col not in df.columns:
        missing = [col for col in (text_col, label_col) if col not in df.columns]
        raise ValueError(f"Missing required columns: {missing}")

    df = df[[text_col, label_col]].dropna().copy()
    df = df.rename(columns={text_col: "text", label_col: "label"})
    before = len(df)
    df = df[df["label"].isin([0, 1])]
    after = len(df)

    if after == 0:
        raise ValueError("No rows remain after filtering (labels must be 0 or 1).")

    if after < before:
        logger.info("Filtered out %d rows with invalid labels", before - after)

    class_counts = df["label"].value_counts().reindex([0, 1], fill_value=0)
    logger.info("Using %d samples (0=%d, 1=%d)", len(df), class_counts[0], class_counts[1])
    return df.reset_index(drop=True)


def load_labeled_data(
    csv_path: str,
    text_col: str = DEFAULT_TEXT_COL,
    label_col: str = DEFAULT_LABEL_COL,
) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Labeled CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info("Loaded data from %s", csv_path)
    return clean_dataframe(df, text_col, label_col)


def train_logistic_regression(
    c_value: float,
    *,
    dataframe: Optional[pd.DataFrame] = None,
    data_path: Optional[str] = None,
    text_col: str = DEFAULT_TEXT_COL,
    label_col: str = DEFAULT_LABEL_COL,
    max_features: int = DEFAULT_MAX_FEATURES,
    ngram_range: Tuple[int, int] = DEFAULT_NGRAM_RANGE,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    cv_folds: int = DEFAULT_CV_FOLDS,
    artifact_dir: Optional[str] = None,
    save_artifacts: bool = False,
    metrics_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    if c_value <= 0:
        raise ValueError("C must be positive")

    if test_size <= 0 or test_size >= 1:
        raise ValueError("test_size must be between 0 and 1")

    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2")

    if dataframe is None:
        if not data_path:
            raise ValueError("Provide either a dataframe or a data_path")
        df = load_labeled_data(data_path, text_col, label_col)
    else:
        df = clean_dataframe(dataframe.copy(), text_col, label_col)

    if save_artifacts and artifact_dir is None:
        artifact_dir = DEFAULT_OUTPUT_DIR

    if artifact_dir:
        ensure_directory(artifact_dir)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)),
        ("logreg", LogisticRegression(C=c_value, random_state=random_state, max_iter=1000)),
    ])

    cv_start = time.time()
    cv_results = cross_validate(
        pipeline,
        df["text"],
        df["label"],
        cv=cv_folds,
        scoring=["accuracy", "precision", "recall", "f1"],
        n_jobs=1,
        return_train_score=False,
        error_score="raise",
    )
    cv_duration = time.time() - cv_start

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(
        C=c_value,
        random_state=random_state,
        max_iter=1000,
        n_jobs=-1,
    )

    train_start = time.time()
    model.fit(X_train_tfidf, y_train)
    training_duration = time.time() - train_start

    y_pred = model.predict(X_test_tfidf)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred, zero_division=0)
    test_recall = recall_score(y_test, y_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_pred, zero_division=0)

    if verbose:
        logger.info(
            "C=%s | test acc=%.4f prec=%.4f rec=%.4f f1=%.4f",
            c_value,
            test_accuracy,
            test_precision,
            test_recall,
            test_f1,
        )

    suffix = sanitize_c_value(c_value)
    model_path = None
    vectorizer_path = None

    if save_artifacts and artifact_dir:
        model_path = os.path.join(artifact_dir, f"logreg_model_c{c_value}_mf{max_features}_ng{ngram_range[0]}-{ngram_range[1]}.pkl")
        vectorizer_path = os.path.join(artifact_dir, f"tfidf_vectorizer_c{c_value}_mf{max_features}_ng{ngram_range[0]}-{ngram_range[1]}.pkl")
        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)
        logger.info("Saved model to %s", model_path)
        logger.info("Saved vectorizer to %s", vectorizer_path)

    if metrics_path is None and save_artifacts and artifact_dir:
        metrics_path = os.path.join(artifact_dir, f"metrics_{suffix}.json")

    metrics = {
        "c_value": c_value,
        "timestamp": datetime.now().isoformat(),
        "data_path": data_path or "<in-memory>",
        "text_column": text_col,
        "label_column": label_col,
        "max_features": max_features,
        "ngram_range": ngram_range,
        "test_size": test_size,
        "random_state": random_state,
        "cv_folds": cv_folds,
        "n_samples": len(df),
        "validation_accuracy_mean": float(np.mean(cv_results["test_accuracy"])),
        "validation_accuracy_std": float(np.std(cv_results["test_accuracy"])),
        "validation_precision_mean": float(np.mean(cv_results["test_precision"])),
        "validation_precision_std": float(np.std(cv_results["test_precision"])),
        "validation_recall_mean": float(np.mean(cv_results["test_recall"])),
        "validation_recall_std": float(np.std(cv_results["test_recall"])),
        "validation_f1_mean": float(np.mean(cv_results["test_f1"])),
        "validation_f1_std": float(np.std(cv_results["test_f1"])),
        "cv_duration": cv_duration,
        "training_duration": training_duration,
        "test_accuracy": test_accuracy,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "model_path": model_path,
        "vectorizer_path": vectorizer_path,
        "metrics_path": metrics_path,
    }

    if metrics_path is not None:
        try:
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            logger.info("Saved metrics to %s", metrics_path)
            metrics["metrics_path"] = metrics_path
        except (OSError, TypeError) as exc:
            logger.warning("Unable to save metrics file (%s): %s", metrics_path, exc)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Logistic Regression models for IndoHoaxDetector."
    )
    parser.add_argument("--c_value", type=float, required=True, help="Regularization strength")
    parser.add_argument("--max_features", type=int, default=DEFAULT_MAX_FEATURES)
    parser.add_argument("--ngram_min", type=int, default=DEFAULT_NGRAM_RANGE[0])
    parser.add_argument("--ngram_max", type=int, default=DEFAULT_NGRAM_RANGE[1])
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--text_col", type=str, default=DEFAULT_TEXT_COL)
    parser.add_argument("--label_col", type=str, default=DEFAULT_LABEL_COL)
    parser.add_argument("--test_size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--cv_folds", type=int, default=DEFAULT_CV_FOLDS)
    parser.add_argument("--random_state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip_artifacts", action="store_true")
    args = parser.parse_args()

    ensure_directory(args.output_dir)

    metrics = train_logistic_regression(
        args.c_value,
        data_path=args.data_path,
        text_col=args.text_col,
        label_col=args.label_col,
        max_features=args.max_features,
        ngram_range=(args.ngram_min, args.ngram_max),
        test_size=args.test_size,
        random_state=args.random_state,
        cv_folds=args.cv_folds,
        artifact_dir=args.output_dir,
        save_artifacts=not args.skip_artifacts,
        metrics_path=os.path.join(args.output_dir, f"metrics_c{args.c_value}_mf{args.max_features}_ng{args.ngram_min}-{args.ngram_max}.json"),
    )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
