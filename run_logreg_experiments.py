#!/usr/bin/env python3
"""Run Logistic Regression C-sweep experiments and summarize results."""

from __future__ import annotations

import argparse
import os
import textwrap
import time
from datetime import datetime
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from train_logreg import (
    DEFAULT_CV_FOLDS,
    DEFAULT_DATA_PATH,
    DEFAULT_LABEL_COL,
    DEFAULT_MAX_FEATURES,
    DEFAULT_NGRAM_RANGE,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    DEFAULT_TEXT_COL,
    load_labeled_data,
    sanitize_c_value,
    train_logistic_regression,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_BASE = os.path.join(SCRIPT_DIR, "results", "logreg_c_experiments")
DEFAULT_C_VALUES = [0.01, 0.1, 1.0, 10.0]


def create_run_directory(base_dir: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"logreg_c_sweep_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def run_c_sweep(
    c_values: List[float],
    df: pd.DataFrame,
    results_dir: str,
    max_features: int,
    ngram_min: int,
    ngram_max: int,
    test_size: float,
    cv_folds: int,
    random_state: int,
    save_artifacts: bool,
    verbose: bool,
) -> Tuple[str, pd.DataFrame]:
    run_dir = create_run_directory(results_dir)
    artifact_base = os.path.join(run_dir, "artifacts")
    os.makedirs(artifact_base, exist_ok=True)

    records = []
    for c_value in c_values:
        suffix = sanitize_c_value(c_value)
        artifact_dir = os.path.join(artifact_base, f"c_{suffix}")
        os.makedirs(artifact_dir, exist_ok=True)

        start_time = time.time()
        metrics = train_logistic_regression(
            c_value,
            dataframe=df,
            max_features=max_features,
            ngram_range=(ngram_min, ngram_max),
            test_size=test_size,
            random_state=random_state,
            cv_folds=cv_folds,
            artifact_dir=artifact_dir,
            save_artifacts=save_artifacts,
            metrics_path=os.path.join(artifact_dir, f"metrics_{suffix}.json"),
            verbose=verbose,
        )
        duration = time.time() - start_time

        metrics.update({
            "artifact_dir": artifact_dir,
            "experiment_start": datetime.fromtimestamp(start_time).isoformat(),
            "experiment_duration": duration,
            "ngram_range": f"{ngram_min}-{ngram_max}",
        })
        records.append(metrics)

    results_df = pd.DataFrame(records)
    return run_dir, results_df


def save_results_csv(results_df: pd.DataFrame, output_dir: str) -> Tuple[str, str]:
    detailed_path = os.path.join(output_dir, "logreg_experiment_results.csv")
    results_df.to_csv(detailed_path, index=False)

    summary_path = os.path.join(output_dir, "logreg_experiment_summary.csv")
    results_df.sort_values("test_f1", ascending=False).to_csv(summary_path, index=False)

    return detailed_path, summary_path


def plot_metrics(results_df: pd.DataFrame, output_dir: str) -> str:
    if results_df.empty:
        return ""

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df = results_df.sort_values("c_value")
    metrics = ["test_accuracy", "test_precision", "test_recall", "test_f1"]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    for metric, color in zip(metrics, colors):
        ax.plot(
            plot_df["c_value"],
            plot_df[metric],
            marker="o",
            color=color,
            label=metric.replace("test_", ""),
        )

    ax.set_xscale("log")
    ax.set_xlabel("C (log scale)")
    ax.set_ylabel("Metric")
    ax.set_title("Logistic Regression Test Metrics vs C")
    ax.grid(alpha=0.3)
    ax.legend()

    plot_path = os.path.join(output_dir, "logreg_metrics_vs_c.png")
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return plot_path


def write_summary_markdown(results_df: pd.DataFrame, output_dir: str) -> str:
    best_row = results_df.loc[results_df["test_f1"].idxmax()]
    summary_path = os.path.join(output_dir, "logreg_experiment_summary.md")

    with open(summary_path, "w") as fp:
        fp.write("# Logistic Regression C Sweep Summary\n\n")
        fp.write(f"**Runtime:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        fp.write("## Dataset and Experimental Setup\n")
        fp.write(f"- Samples: {int(best_row['n_samples'])}\n")
        fp.write(f"- Test split size: {best_row['test_size']}\n")
        fp.write(f"- CV folds: {int(best_row['cv_folds'])}\n")
        fp.write(f"- TF-IDF: max_features={best_row['max_features']}, ngram_range={best_row['ngram_range']}\n\n")

        fp.write("## Best C Value\n")
        fp.write(f"- **C = {best_row['c_value']}** (Test F1 = {best_row['test_f1']:.4f})\n")
        fp.write(f"- Test Accuracy: {best_row['test_accuracy']:.4f}\n")
        fp.write(f"- Test Precision: {best_row['test_precision']:.4f}\n")
        fp.write(f"- Test Recall: {best_row['test_recall']:.4f}\n\n")

        fp.write("## Metrics by C\n")
        fp.write("| C | Test Accuracy | Precision | Recall | F1 | Duration (s) |\n")
        fp.write("|---|---------------|-----------|--------|----|--------------|\n")
        for _, row in results_df.sort_values("c_value").iterrows():
            fp.write(
                f"| {row['c_value']} | {row['test_accuracy']:.4f} | {row['test_precision']:.4f} | "
                f"{row['test_recall']:.4f} | {row['test_f1']:.4f} | {row['experiment_duration']:.2f} |\n"
            )

        fp.write("\n## Recommendations\n")
        fp.write("- Deploy the model trained with the highest test F1 (above).\n")
        fp.write("- Extend the sweep if you collect more data or want to explore smaller/greater C values.\n")

    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Logistic Regression experiments sweeping C values."
    )
    parser.add_argument("--c_values", type=float, nargs="+", default=DEFAULT_C_VALUES)
    parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_BASE)
    parser.add_argument("--max_features", type=int, default=DEFAULT_MAX_FEATURES)
    parser.add_argument("--ngram_min", type=int, default=DEFAULT_NGRAM_RANGE[0])
    parser.add_argument("--ngram_max", type=int, default=DEFAULT_NGRAM_RANGE[1])
    parser.add_argument("--test_size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--cv_folds", type=int, default=DEFAULT_CV_FOLDS)
    parser.add_argument("--random_state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--text_col", type=str, default=DEFAULT_TEXT_COL)
    parser.add_argument("--label_col", type=str, default=DEFAULT_LABEL_COL)
    parser.add_argument("--skip_plot", action="store_true")
    parser.add_argument("--no_artifacts", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if any(c <= 0 for c in args.c_values):
        raise SystemExit("C values must be positive.")

    df = load_labeled_data(args.data_path, args.text_col, args.label_col)
    run_dir, results_df = run_c_sweep(
        args.c_values,
        df,
        args.results_dir,
        args.max_features,
        args.ngram_min,
        args.ngram_max,
        args.test_size,
        args.cv_folds,
        args.random_state,
        not args.no_artifacts,
        args.verbose,
    )

    detailed_csv, summary_csv = save_results_csv(results_df, run_dir)
    plot_path = ""
    if not args.skip_plot:
        plot_path = plot_metrics(results_df, run_dir)
    summary_md = write_summary_markdown(results_df, run_dir)

    print(textwrap.dedent(f"""
        Experiments complete!
        Run directory: {run_dir}
        Detailed CSV: {detailed_csv}
        Summary CSV: {summary_csv}
        Summary Markdown: {summary_md}
        Plot: {plot_path or 'skipped'}
    """))

    best_row = results_df.loc[results_df["test_f1"].idxmax()]
    print("Best C", best_row["c_value"], "with test F1", f"{best_row['test_f1']:.4f}")


if __name__ == "__main__":
    main()
