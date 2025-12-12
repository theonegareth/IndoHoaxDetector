#!/usr/bin/env python3
"""
Run Logistic Regression hyperparameter experiments for IndoHoaxDetector.

This script runs train_logreg.py for multiple C values, collects results,
and generates summary visualizations and reports.

Usage:
    python run_logreg_experiments.py
    python run_logreg_experiments.py --c_values 0.01 0.1 1.0 10.0 100.0
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# CONFIGURATION
# =========================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_C_VALUES = [0.01, 0.1, 1.0, 10.0, 100.0]
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results")
TRAIN_SCRIPT = os.path.join(SCRIPT_DIR, "train_logreg.py")

# =========================
# EXPERIMENT EXECUTION
# =========================

def run_single_experiment(c_value: float, output_dir: str, max_features: int, ngram_min: int, ngram_max: int) -> Dict[str, Any]:
    """Run a single experiment with the specified C value and TF-IDF parameters."""
    
    print(f"\n{'='*60}")
    print(f"RUNNING EXPERIMENT: C = {c_value}, max_features={max_features}, ngram_range=({ngram_min},{ngram_max})")
    print(f"{'='*60}")
    
    # Prepare command
    cmd = [
        sys.executable,
        TRAIN_SCRIPT,
        "--c_value", str(c_value),
        "--max_features", str(max_features),
        "--ngram_min", str(ngram_min),
        "--ngram_max", str(ngram_max),
        "--output_dir", output_dir
    ]
    
    # Run the training script
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        success = True
        stdout = result.stdout
        stderr = result.stderr
    except subprocess.CalledProcessError as e:
        success = False
        stdout = e.stdout
        stderr = e.stderr
        print(f"[ERROR] Experiment failed for C={c_value}: {e}")
    
    experiment_duration = time.time() - start_time
    
    # Try to load metrics if successful
    metrics = None
    if success:
        metrics_path = os.path.join(output_dir, f"metrics_c{c_value}.json")
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
        except FileNotFoundError:
            print(f"[WARNING] Metrics file not found: {metrics_path}")
        except json.JSONDecodeError as e:
            print(f"[WARNING] Failed to parse metrics: {e}")
    
    return {
        'c_value': c_value,
        'success': success,
        'experiment_duration': experiment_duration,
        'timestamp': datetime.now().isoformat(),
        'stdout': stdout,
        'stderr': stderr,
        'metrics': metrics
    }

def run_experiments(c_values: List[float], output_dir: str, max_features: int, ngram_min: int, ngram_max: int) -> List[Dict[str, Any]]:
    """Run experiments for all C values with given TF-IDF parameters."""
    
    print(f"[INFO] Starting experiments for {len(c_values)} C values: {c_values}")
    print(f"[INFO] TF-IDF parameters: max_features={max_features}, ngram_range=({ngram_min},{ngram_max})")
    print(f"[INFO] Output directory: {output_dir}")
    
    results = []
    for c_value in c_values:
        result = run_single_experiment(c_value, output_dir, max_features, ngram_min, ngram_max)
        results.append(result)
        
        # Print summary for this experiment
        if result['success'] and result['metrics']:
            m = result['metrics']
            print(f"[INFO] C={c_value}: Acc={m['accuracy_mean']:.4f}±{m['accuracy_std']:.4f}, "
                  f"F1={m['f1_mean']:.4f}±{m['f1_std']:.4f}")
        else:
            print(f"[INFO] C={c_value}: FAILED")
    
    return results

# =========================
# RESULTS PROCESSING
# =========================

def process_results(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Process experiment results into a DataFrame."""
    
    processed_data = []
    
    for result in results:
        if not result['success'] or not result['metrics']:
            # Include failed experiments with NaN values
            processed_data.append({
                'c_value': result['c_value'],
                'success': False,
                'experiment_duration': result['experiment_duration'],
                'timestamp': result['timestamp'],
                'accuracy_mean': float('nan'),
                'accuracy_std': float('nan'),
                'precision_mean': float('nan'),
                'precision_std': float('nan'),
                'recall_mean': float('nan'),
                'recall_std': float('nan'),
                'f1_mean': float('nan'),
                'f1_std': float('nan'),
                'training_duration': float('nan'),
                'cv_folds': float('nan'),
                'n_samples': float('nan')
            })
            continue
        
        m = result['metrics']
        processed_data.append({
            'c_value': m['c_value'],
            'success': True,
            'experiment_duration': result['experiment_duration'],
            'timestamp': result['timestamp'],
            'accuracy_mean': m['accuracy_mean'],
            'accuracy_std': m['accuracy_std'],
            'precision_mean': m['precision_mean'],
            'precision_std': m['precision_std'],
            'recall_mean': m['recall_mean'],
            'recall_std': m['recall_std'],
            'f1_mean': m['f1_mean'],
            'f1_std': m['f1_std'],
            'training_duration': m['training_duration'],
            'cv_folds': m['cv_folds'],
            'n_samples': m['n_samples']
        })
    
    return pd.DataFrame(processed_data)

def save_results(results_df: pd.DataFrame, output_dir: str):
    """Save results to CSV and other formats."""
    
    # Save detailed results
    csv_path = os.path.join(output_dir, "experiment_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"[INFO] Saved detailed results to: {csv_path}")
    
    # Save summary (successful experiments only)
    successful_df = results_df[results_df['success']].copy()
    if not successful_df.empty:
        # Sort by F1 score for summary
        summary_df = successful_df.sort_values('f1_mean', ascending=False)
        summary_path = os.path.join(output_dir, "experiment_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"[INFO] Saved summary to: {summary_path}")
    
    return csv_path

# =========================
# VISUALIZATION
# =========================

def create_visualizations(results_df: pd.DataFrame, output_dir: str):
    """Create summary plots of the results."""
    
    # Filter successful experiments
    successful_df = results_df[results_df['success']].copy()
    if successful_df.empty:
        print("[WARNING] No successful experiments to visualize.")
        return
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Logistic Regression Hyperparameter Tuning Results', fontsize=16, fontweight='bold')
    
    # Sort by C value for plotting
    plot_df = successful_df.sort_values('c_value')
    
    # Accuracy plot
    axes[0, 0].errorbar(plot_df['c_value'], plot_df['accuracy_mean'], 
                       yerr=plot_df['accuracy_std'], marker='o', capsize=5,
                       label='Accuracy', color='blue')
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_xlabel('C Value (log scale)')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Accuracy vs C Value')
    axes[0, 0].grid(True, alpha=0.3)
    
    # F1 Score plot
    axes[0, 1].errorbar(plot_df['c_value'], plot_df['f1_mean'], 
                       yerr=plot_df['f1_std'], marker='s', capsize=5,
                       label='F1 Score', color='green')
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_xlabel('C Value (log scale)')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_title('F1 Score vs C Value')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision vs Recall
    axes[1, 0].errorbar(plot_df['c_value'], plot_df['precision_mean'], 
                       yerr=plot_df['precision_std'], marker='^', capsize=5,
                       label='Precision', color='red')
    axes[1, 0].errorbar(plot_df['c_value'], plot_df['recall_mean'], 
                       yerr=plot_df['recall_std'], marker='v', capsize=5,
                       label='Recall', color='orange')
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_xlabel('C Value (log scale)')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Precision vs Recall vs C Value')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Training time
    axes[1, 1].plot(plot_df['c_value'], plot_df['training_duration'], 
                   marker='d', color='purple', linewidth=2)
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_xlabel('C Value (log scale)')
    axes[1, 1].set_ylabel('Training Time (seconds)')
    axes[1, 1].set_title('Training Time vs C Value')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, "hyperparameter_tuning_results.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Saved visualization to: {plot_path}")
    
    plt.close()

# =========================
# REPORTING
# =========================

def generate_report(results_df: pd.DataFrame, output_dir: str, max_features: int, ngram_min: int, ngram_max: int):
    """Generate a comprehensive experiment report with TF-IDF parameters."""
    
    successful_df = results_df[results_df['success']].copy()
    
    report_path = os.path.join(output_dir, "experiment_report.md")
    
    with open(report_path, 'w') as f:
        f.write("# Logistic Regression Hyperparameter Tuning Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**TF-IDF Parameters:** max_features={max_features}, ngram_range=({ngram_min},{ngram_max})\n\n")
        
        f.write("## Experiment Summary\n\n")
        f.write(f"- **Total Experiments:** {len(results_df)}\n")
        f.write(f"- **Successful Experiments:** {len(successful_df)}\n")
        f.write(f"- **Failed Experiments:** {len(results_df) - len(successful_df)}\n")
        
        if not successful_df.empty:
            f.write(f"- **Dataset Size:** {int(successful_df['n_samples'].iloc[0])} samples\n")
            f.write(f"- **Cross-validation Folds:** {int(successful_df['cv_folds'].iloc[0])}\n")
        
        f.write("\n## Results Overview\n\n")
        
        if successful_df.empty:
            f.write("❌ **All experiments failed.** Check the logs for details.\n\n")
        else:
            # Best performing model
            best_f1 = successful_df.loc[successful_df['f1_mean'].idxmax()]
            best_acc = successful_df.loc[successful_df['accuracy_mean'].idxmax()]
            
            f.write("### Best Performing Models\n\n")
            f.write("#### By F1 Score\n")
            f.write(f"- **C Value:** {best_f1['c_value']}\n")
            f.write(f"- **F1 Score:** {best_f1['f1_mean']:.4f} ± {best_f1['f1_std']:.4f}\n")
            f.write(f"- **Accuracy:** {best_f1['accuracy_mean']:.4f} ± {best_f1['accuracy_std']:.4f}\n")
            f.write(f"- **Precision:** {best_f1['precision_mean']:.4f} ± {best_f1['precision_std']:.4f}\n")
            f.write(f"- **Recall:** {best_f1['recall_mean']:.4f} ± {best_f1['recall_std']:.4f}\n")
            f.write(f"- **Training Time:** {best_f1['training_duration']:.2f} seconds\n\n")
            
            f.write("#### By Accuracy\n")
            f.write(f"- **C Value:** {best_acc['c_value']}\n")
            f.write(f"- **Accuracy:** {best_acc['accuracy_mean']:.4f} ± {best_acc['accuracy_std']:.4f}\n")
            f.write(f"- **F1 Score:** {best_acc['f1_mean']:.4f} ± {best_acc['f1_std']:.4f}\n\n")
            
            # Performance comparison table
            f.write("### Performance Comparison\n\n")
            f.write("| C Value | Accuracy | Precision | Recall | F1 Score | Training Time |\n")
            f.write("|--------|----------|-----------|--------|----------|---------------|\n")
            
            for _, row in successful_df.sort_values('c_value').iterrows():
                f.write(f"| {row['c_value']} | {row['accuracy_mean']:.4f}±{row['accuracy_std']:.4f} | ")
                f.write(f"{row['precision_mean']:.4f}±{row['precision_std']:.4f} | ")
                f.write(f"{row['recall_mean']:.4f}±{row['recall_std']:.4f} | ")
                f.write(f"{row['f1_mean']:.4f}±{row['f1_std']:.4f} | ")
                f.write(f"{row['training_duration']:.2f}s |\n")
            
            f.write("\n## Recommendations\n\n")
            
            # Analyze the results
            c_values = successful_df['c_value'].values
            f1_scores = successful_df['f1_mean'].values
            
            # Find optimal C
            optimal_idx = np.argmax(f1_scores)
            optimal_c = c_values[optimal_idx]
            
            f.write(f"### Optimal Hyperparameter\n\n")
            f.write(f"Based on F1 score performance, **C = {optimal_c}** appears to be the optimal regularization parameter.\n\n")
            
            # Check for overfitting/underfitting patterns
            if len(c_values) > 1:
                # Simple trend analysis
                if f1_scores[0] < f1_scores[-1]:  # Performance improves with higher C
                    f.write("### Regularization Analysis\n\n")
                    f.write("Performance tends to improve with higher C values, suggesting that lower regularization ")
                    f.write("(higher C) is beneficial for this dataset. This indicates the model may have been ")
                    f.write("under-regularized with very small C values.\n\n")
                elif f1_scores[0] > f1_scores[-1]:  # Performance decreases with higher C
                    f.write("### Regularization Analysis\n\n")
                    f.write("Performance tends to decrease with higher C values, suggesting that higher regularization ")
                    f.write("(lower C) is beneficial for this dataset. This indicates the model may have been ")
                    f.write("overfitting with very high C values.\n\n")
                else:
                    f.write("### Regularization Analysis\n\n")
                    f.write("Performance shows mixed results across C values. The optimal C value provides ")
                    f.write("the best balance between bias and variance for this dataset.\n\n")
            
            f.write("### Usage Recommendations\n\n")
            f.write(f"1. **Use C = {optimal_c}** for production deployment\n")
            f.write("2. Monitor model performance on new data regularly\n")
            f.write("3. Consider retraining with more data if performance is insufficient\n")
            f.write("4. Evaluate on additional metrics relevant to your use case\n\n")
        
        f.write("## Files Generated\n\n")
        f.write("- `experiment_results.csv`: Detailed results for all experiments\n")
        f.write("- `experiment_summary.csv`: Summary of successful experiments\n")
        f.write("- `hyperparameter_tuning_results.png`: Performance visualization\n")
        f.write("- Individual model files: `logreg_model_c{value}.pkl`\n")
        f.write("- Individual vectorizers: `tfidf_vectorizer_c{value}.pkl`\n")
        f.write("- Individual metrics: `metrics_c{value}.json`\n\n")
        
        f.write("## Technical Details\n\n")
        f.write("- **Model:** LogisticRegression (sklearn)\n")
        f.write(f"- **Features:** TF-IDF vectorization (max_features={max_features}, ngram_range=({ngram_min},{ngram_max}))\n")
        f.write("- **Evaluation:** 5-fold stratified cross-validation\n")
        f.write("- **Metrics:** Accuracy, Precision, Recall, F1 Score\n")
        f.write("- **Random State:** 42 (for reproducibility)\n")
    
    print(f"[INFO] Generated report: {report_path}")
    return report_path

# =========================
# MAIN
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="Run Logistic Regression hyperparameter experiments for hoax detection."
    )
    parser.add_argument(
        "--c_values",
        type=float,
        nargs='+',
        default=DEFAULT_C_VALUES,
        help=f"C values to test (default: {DEFAULT_C_VALUES})"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for results (default: {DEFAULT_OUTPUT_DIR})"
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
        "--skip_plots",
        action='store_true',
        help="Skip generating visualization plots"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.c_values:
        print("[ERROR] No C values specified.", file=sys.stderr)
        sys.exit(1)
    
    for c in args.c_values:
        if c <= 0:
            print(f"[ERROR] C values must be positive. Got: {c}", file=sys.stderr)
            sys.exit(1)
    
    # Check if train script exists
    if not os.path.exists(TRAIN_SCRIPT):
        print(f"[ERROR] Training script not found: {TRAIN_SCRIPT}", file=sys.stderr)
        sys.exit(1)
    
    print(f"[INFO] Starting hyperparameter tuning experiments")
    print(f"[INFO] C values to test: {args.c_values}")
    print(f"[INFO] TF-IDF parameters: max_features={args.max_features}, ngram_range=({args.ngram_min},{args.ngram_max})")
    print(f"[INFO] Output directory: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run experiments
    try:
        results = run_experiments(
            args.c_values,
            args.output_dir,
            args.max_features,
            args.ngram_min,
            args.ngram_max
        )
        
        # Process and save results
        results_df = process_results(results)
        csv_path = save_results(results_df, args.output_dir)
        
        # Generate visualizations
        if not args.skip_plots:
            create_visualizations(results_df, args.output_dir)
        
        # Generate report
        report_path = generate_report(results_df, args.output_dir, args.max_features, args.ngram_min, args.ngram_max)
        
        print(f"\n{'='*60}")
        print("EXPERIMENTS COMPLETED")
        print(f"{'='*60}")
        print(f"Results saved to: {args.output_dir}")
        print(f"Detailed CSV: {csv_path}")
        print(f"Report: {report_path}")
        
        # Print best result summary
        successful_df = results_df[results_df['success']]
        if not successful_df.empty:
            best_result = successful_df.loc[successful_df['f1_mean'].idxmax()]
            print(f"\nBest C value: {best_result['c_value']}")
            print(f"Best F1 Score: {best_result['f1_mean']:.4f} ± {best_result['f1_std']:.4f}")
        
    except Exception as e:
        print(f"[ERROR] Experiments failed: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()