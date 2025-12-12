#!/usr/bin/env python3
"""
Run hyperparameter experiments for all models in IndoHoaxDetector.

This script runs experiments for Logistic Regression, SVM, Random Forest,
and Naive Bayes with their respective hyperparameters, collects results,
and generates comprehensive comparisons.

Usage:
    python run_all_experiments.py
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
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results")

# Hyperparameter configurations for each model
EXPERIMENT_CONFIGS = {
    "Logistic Regression": {
        "script": os.path.join(SCRIPT_DIR, "train_logreg.py"),
        "param_name": "c_value",
        "param_values": [0.01, 0.1, 1.0, 10.0, 100.0],
        "param_flag": "--c_value"
    },
    "Linear SVM": {
        "script": os.path.join(SCRIPT_DIR, "train_svm.py"),
        "param_name": "c_value",
        "param_values": [0.01, 0.1, 1.0, 10.0, 100.0],
        "param_flag": "--c_value"
    },
    "Random Forest": {
        "script": os.path.join(SCRIPT_DIR, "train_rf.py"),
        "param_name": "n_estimators",
        "param_values": [50, 100, 200, 500],
        "param_flag": "--n_estimators"
    },
    "Multinomial Naive Bayes": {
        "script": os.path.join(SCRIPT_DIR, "train_nb.py"),
        "param_name": "alpha",
        "param_values": [0.1, 0.5, 1.0, 2.0, 5.0],
        "param_flag": "--alpha"
    }
}

# =========================
# EXPERIMENT EXECUTION
# =========================

def run_single_experiment(
    model_name: str,
    script_path: str,
    param_flag: str,
    param_value: Any,
    output_dir: str,
    max_features: int,
    ngram_min: int,
    ngram_max: int
) -> Dict[str, Any]:
    """Run a single experiment for a specific model and parameter value."""
    
    print(f"\n{'='*60}")
    print(f"RUNNING EXPERIMENT: {model_name} - {param_flag} {param_value}")
    print(f"TF-IDF: max_features={max_features}, ngram_range=({ngram_min},{ngram_max})")
    print(f"{'='*60}")
    
    # Prepare command
    cmd = [
        sys.executable,
        script_path,
        param_flag, str(param_value),
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
        print(f"[ERROR] Experiment failed for {model_name} with {param_flag}={param_value}: {e}")
    
    experiment_duration = time.time() - start_time
    
    # Try to load metrics if successful
    metrics = None
    if success:
        # Determine metrics file pattern based on model
        if model_name == "Logistic Regression":
            metrics_file = f"metrics_c{param_value}.json"
        elif model_name == "Linear SVM":
            metrics_file = f"svm_metrics_c{param_value}.json"
        elif model_name == "Random Forest":
            metrics_file = f"rf_metrics_n{param_value}.json"
        elif model_name == "Multinomial Naive Bayes":
            metrics_file = f"nb_metrics_a{param_value}.json"
        else:
            metrics_file = None
        
        if metrics_file:
            metrics_path = os.path.join(output_dir, metrics_file)
            try:
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
            except FileNotFoundError:
                print(f"[WARNING] Metrics file not found: {metrics_path}")
            except json.JSONDecodeError as e:
                print(f"[WARNING] Failed to parse metrics: {e}")
    
    return {
        'model_name': model_name,
        'param_name': param_flag.replace('--', ''),
        'param_value': param_value,
        'success': success,
        'experiment_duration': experiment_duration,
        'timestamp': datetime.now().isoformat(),
        'stdout': stdout,
        'stderr': stderr,
        'metrics': metrics
    }

def run_all_experiments(output_dir: str, max_features: int, ngram_min: int, ngram_max: int) -> List[Dict[str, Any]]:
    """Run experiments for all models and their hyperparameters."""
    
    print(f"[INFO] Starting comprehensive model experiments")
    print(f"[INFO] TF-IDF parameters: max_features={max_features}, ngram_range=({ngram_min},{ngram_max})")
    print(f"[INFO] Output directory: {output_dir}")
    
    results = []
    
    for model_name, config in EXPERIMENT_CONFIGS.items():
        print(f"\n{'='*80}")
        print(f"STARTING EXPERIMENTS FOR: {model_name}")
        print(f"{'='*80}")
        
        script_path = config["script"]
        param_flag = config["param_flag"]
        param_values = config["param_values"]
        
        # Check if script exists
        if not os.path.exists(script_path):
            print(f"[WARNING] Script not found: {script_path}. Skipping {model_name}.")
            continue
        
        for param_value in param_values:
            result = run_single_experiment(
                model_name, script_path, param_flag, param_value, output_dir,
                max_features, ngram_min, ngram_max
            )
            results.append(result)
            
            # Print summary for this experiment
            if result['success'] and result['metrics']:
                m = result['metrics']
                print(f"[INFO] {model_name} ({param_value}): Acc={m['accuracy_mean']:.4f}±{m['accuracy_std']:.4f}, "
                      f"F1={m['f1_mean']:.4f}±{m['f1_std']:.4f}")
            else:
                print(f"[INFO] {model_name} ({param_value}): FAILED")
    
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
                'model_name': result['model_name'],
                'param_name': result['param_name'],
                'param_value': result['param_value'],
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
            'model_name': result['model_name'],
            'param_name': result['param_name'],
            'param_value': result['param_value'],
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
            'training_duration': m.get('training_duration', float('nan')),
            'cv_folds': m.get('cv_folds', float('nan')),
            'n_samples': m.get('n_samples', float('nan'))
        })
    
    return pd.DataFrame(processed_data)

def save_results(results_df: pd.DataFrame, output_dir: str):
    """Save results to CSV and other formats."""
    
    # Save detailed results
    csv_path = os.path.join(output_dir, "all_models_experiment_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"[INFO] Saved detailed results to: {csv_path}")
    
    # Save summary (successful experiments only)
    successful_df = results_df[results_df['success']].copy()
    if not successful_df.empty:
        # Sort by F1 score for summary
        summary_df = successful_df.sort_values('f1_mean', ascending=False)
        summary_path = os.path.join(output_dir, "all_models_experiment_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"[INFO] Saved summary to: {summary_path}")
    
    return csv_path

# =========================
# VISUALIZATION
# =========================

def create_comprehensive_visualizations(results_df: pd.DataFrame, output_dir: str):
    """Create comprehensive visualizations comparing all models."""
    
    # Filter successful experiments
    successful_df = results_df[results_df['success']].copy()
    if successful_df.empty:
        print("[WARNING] No successful experiments to visualize.")
        return
    
    # Set style
    sns.set_style("whitegrid")
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comprehensive Model Comparison - IndoHoaxDetector', fontsize=16, fontweight='bold')
    
    # 1. F1 Score comparison across models and parameters
    ax1 = axes[0, 0]
    for model_name in successful_df['model_name'].unique():
        model_data = successful_df[successful_df['model_name'] == model_name]
        ax1.errorbar(model_data['param_value'], model_data['f1_mean'], 
                    yerr=model_data['f1_std'], marker='o', capsize=3,
                    label=model_name, linewidth=2)
    ax1.set_xlabel('Hyperparameter Value')
    ax1.set_ylabel('F1 Score')
    ax1.set_title('F1 Score by Model and Hyperparameter')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Accuracy comparison
    ax2 = axes[0, 1]
    for model_name in successful_df['model_name'].unique():
        model_data = successful_df[successful_df['model_name'] == model_name]
        ax2.errorbar(model_data['param_value'], model_data['accuracy_mean'], 
                    yerr=model_data['accuracy_std'], marker='s', capsize=3,
                    label=model_name, linewidth=2)
    ax2.set_xlabel('Hyperparameter Value')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy by Model and Hyperparameter')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # 3. Training time comparison
    ax3 = axes[0, 2]
    for model_name in successful_df['model_name'].unique():
        model_data = successful_df[successful_df['model_name'] == model_name]
        ax3.plot(model_data['param_value'], model_data['training_duration'], 
                marker='^', linewidth=2, label=model_name)
    ax3.set_xlabel('Hyperparameter Value')
    ax3.set_ylabel('Training Time (seconds)')
    ax3.set_title('Training Time by Model and Hyperparameter')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # 4. Best F1 scores per model (bar chart)
    ax4 = axes[1, 0]
    best_scores = successful_df.groupby('model_name')['f1_mean'].max()
    best_scores.plot(kind='bar', ax=ax4, color='skyblue', edgecolor='black')
    ax4.set_xlabel('Model')
    ax4.set_ylabel('Best F1 Score')
    ax4.set_title('Best F1 Score per Model')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # 5. Precision vs Recall scatter plot
    ax5 = axes[1, 1]
    for model_name in successful_df['model_name'].unique():
        model_data = successful_df[successful_df['model_name'] == model_name]
        ax5.scatter(model_data['precision_mean'], model_data['recall_mean'], 
                   s=50, alpha=0.7, label=model_name)
    ax5.set_xlabel('Precision')
    ax5.set_ylabel('Recall')
    ax5.set_title('Precision vs Recall by Model')
    ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(0.8, 1.0)
    ax5.set_ylim(0.8, 1.0)
    
    # 6. Model ranking by F1 score
    ax6 = axes[1, 2]
    # Get best configuration for each model
    best_configs = successful_df.loc[successful_df.groupby('model_name')['f1_mean'].idxmax()]
    best_configs = best_configs.sort_values('f1_mean', ascending=True)
    
    bars = ax6.barh(best_configs['model_name'], best_configs['f1_mean'], 
                    color='lightgreen', edgecolor='black')
    ax6.set_xlabel('F1 Score')
    ax6.set_title('Model Ranking (Best F1 Score)')
    ax6.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, best_configs['f1_mean']):
        ax6.text(value + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{value:.4f}', ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, "comprehensive_model_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Saved comprehensive visualization to: {plot_path}")
    
    plt.close()

# =========================
# REPORTING
# =========================

def generate_comprehensive_report(results_df: pd.DataFrame, output_dir: str, max_features: int, ngram_min: int, ngram_max: int):
    """Generate a comprehensive experiment report for all models."""
    
    successful_df = results_df[results_df['success']].copy()
    
    report_path = os.path.join(output_dir, "comprehensive_experiment_report.md")
    
    with open(report_path, 'w') as f:
        f.write("# Comprehensive Model Comparison Report - IndoHoaxDetector\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**TF-IDF Parameters:** max_features={max_features}, ngram_range=({ngram_min},{ngram_max})\n\n")
        
        f.write("## Experiment Summary\n\n")
        f.write(f"- **Total Experiments:** {len(results_df)}\n")
        f.write(f"- **Successful Experiments:** {len(successful_df)}\n")
        f.write(f"- **Failed Experiments:** {len(results_df) - len(successful_df)}\n")
        
        if not successful_df.empty:
            f.write(f"- **Dataset Size:** {int(successful_df['n_samples'].iloc[0])} samples\n")
            f.write(f"- **Cross-validation Folds:** {int(successful_df['cv_folds'].iloc[0])}\n")
            f.write(f"- **Models Tested:** {', '.join(successful_df['model_name'].unique())}\n")
        
        f.write("\n## Model Performance Overview\n\n")
        
        if successful_df.empty:
            f.write("❌ **All experiments failed.** Check the logs for details.\n\n")
        else:
            # Overall best model
            overall_best = successful_df.loc[successful_df['f1_mean'].idxmax()]
            
            f.write("### Overall Best Performing Model\n\n")
            f.write(f"- **Model:** {overall_best['model_name']}\n")
            f.write(f"- **Hyperparameter:** {overall_best['param_name']} = {overall_best['param_value']}\n")
            f.write(f"- **F1 Score:** {overall_best['f1_mean']:.4f} ± {overall_best['f1_std']:.4f}\n")
            f.write(f"- **Accuracy:** {overall_best['accuracy_mean']:.4f} ± {overall_best['accuracy_std']:.4f}\n")
            f.write(f"- **Precision:** {overall_best['precision_mean']:.4f} ± {overall_best['precision_std']:.4f}\n")
            f.write(f"- **Recall:** {overall_best['recall_mean']:.4f} ± {overall_best['recall_std']:.4f}\n")
            f.write(f"- **Training Time:** {overall_best['training_duration']:.2f} seconds\n\n")
            
            # Best configuration per model
            f.write("### Best Configuration per Model\n\n")
            f.write("| Model | Best Hyperparameter | F1 Score | Accuracy | Precision | Recall | Training Time |\n")
            f.write("|-------|-------------------|----------|----------|-----------|--------|---------------|\n")
            
            for model_name in successful_df['model_name'].unique():
                model_data = successful_df[successful_df['model_name'] == model_name]
                best_config = model_data.loc[model_data['f1_mean'].idxmax()]
                
                f.write(f"| {model_name} | {best_config['param_name']}={best_config['param_value']} | ")
                f.write(f"{best_config['f1_mean']:.4f}±{best_config['f1_std']:.4f} | ")
                f.write(f"{best_config['accuracy_mean']:.4f}±{best_config['accuracy_std']:.4f} | ")
                f.write(f"{best_config['precision_mean']:.4f}±{best_config['precision_std']:.4f} | ")
                f.write(f"{best_config['recall_mean']:.4f}±{best_config['recall_std']:.4f} | ")
                f.write(f"{best_config['training_duration']:.2f}s |\n")
            
            f.write("\n## Detailed Performance Comparison\n\n")
            
            # Performance comparison table for all configurations
            f.write("### All Configurations Performance\n\n")
            f.write("| Model | Hyperparameter | F1 Score | Accuracy | Precision | Recall | Training Time |\n")
            f.write("|-------|---------------|----------|----------|-----------|--------|---------------|\n")
            
            for _, row in successful_df.sort_values(['model_name', 'f1_mean'], ascending=[True, False]).iterrows():
                f.write(f"| {row['model_name']} | {row['param_name']}={row['param_value']} | ")
                f.write(f"{row['f1_mean']:.4f}±{row['f1_std']:.4f} | ")
                f.write(f"{row['accuracy_mean']:.4f}±{row['accuracy_std']:.4f} | ")
                f.write(f"{row['precision_mean']:.4f}±{row['precision_std']:.4f} | ")
                f.write(f"{row['recall_mean']:.4f}±{row['recall_std']:.4f} | ")
                f.write(f"{row['training_duration']:.2f}s |\n")
            
            f.write("\n## Analysis and Recommendations\n\n")
            
            # Model comparison analysis
            f.write("### Model Comparison Analysis\n\n")
            
            # Calculate performance statistics
            model_stats = successful_df.groupby('model_name').agg({
                'f1_mean': ['max', 'mean', 'std'],
                'accuracy_mean': ['max', 'mean'],
                'training_duration': ['mean', 'min', 'max']
            }).round(4)
            
            # Find best overall model
            best_model = successful_df.loc[successful_df['f1_mean'].idxmax(), 'model_name']
            f.write(f"**Best Overall Model:** {best_model}\n\n")
            
            # Performance vs speed analysis
            f.write("#### Performance vs Speed Trade-offs\n\n")
            for model_name in successful_df['model_name'].unique():
                model_data = successful_df[successful_df['model_name'] == model_name]
                best_f1 = model_data['f1_mean'].max()
                avg_time = model_data['training_duration'].mean()
                f.write(f"- **{model_name}:** Best F1 = {best_f1:.4f}, Avg Training Time = {avg_time:.2f}s\n")
            
            f.write("\n#### Hyperparameter Sensitivity\n\n")
            for model_name in successful_df['model_name'].unique():
                model_data = successful_df[successful_df['model_name'] == model_name]
                if len(model_data) > 1:
                    f1_range = model_data['f1_mean'].max() - model_data['f1_mean'].min()
                    f.write(f"- **{model_name}:** F1 score range = {f1_range:.4f} "
                           f"(from {model_data['f1_mean'].min():.4f} to {model_data['f1_mean'].max():.4f})\n")
            
            f.write("\n## Recommendations\n\n")
            
            # Production recommendations
            f.write("### Production Deployment Recommendations\n\n")
            
            # Rank models by F1 score
            model_ranking = successful_df.groupby('model_name')['f1_mean'].max().sort_values(ascending=False)
            
            f.write("**Model Ranking by F1 Score:**\n")
            for i, (model, score) in enumerate(model_ranking.items(), 1):
                f.write(f"{i}. **{model}** (F1: {score:.4f})\n")
            
            f.write(f"\n**Recommended Model:** {model_ranking.index[0]}\n")
            
            # Best hyperparameters for each model
            f.write("\n**Optimal Hyperparameters:**\n")
            for model_name in successful_df['model_name'].unique():
                model_data = successful_df[successful_df['model_name'] == model_name]
                best_config = model_data.loc[model_data['f1_mean'].idxmax()]
                param_name = best_config['param_name']
                param_value = best_config['param_value']
                f.write(f"- **{model_name}:** {param_name} = {param_value}\n")
            
            f.write("\n### Usage Guidelines\n\n")
            f.write("1. **Primary Choice:** Use the highest-ranked model for production deployment\n")
            f.write("2. **Speed Considerations:** If inference speed is critical, consider trade-offs between performance and training time\n")
            f.write("3. **Robustness:** All models show good performance; choose based on specific requirements\n")
            f.write("4. **Monitoring:** Regularly evaluate model performance on new data\n")
            f.write("5. **Retraining:** Consider retraining with more recent data periodically\n\n")
        
        f.write("## Files Generated\n\n")
        f.write("- `all_models_experiment_results.csv`: Detailed results for all experiments\n")
        f.write("- `all_models_experiment_summary.csv`: Summary of successful experiments\n")
        f.write("- `comprehensive_model_comparison.png`: Performance visualization\n")
        f.write("- `comprehensive_experiment_report.md`: This report\n")
        f.write("- Individual model files: `{model}_model_{param}{value}.pkl`\n")
        f.write("- Individual vectorizers: `tfidf_vectorizer_{model}_{param}{value}.pkl`\n")
        f.write("- Individual metrics: `{model}_metrics_{param}{value}.json`\n\n")
        
        f.write("## Technical Details\n\n")
        f.write(f"- **Features:** TF-IDF vectorization (max_features={max_features}, ngram_range=({ngram_min},{ngram_max}))\n")
        f.write("- **Evaluation:** 5-fold stratified cross-validation\n")
        f.write("- **Metrics:** Accuracy, Precision, Recall, F1 Score\n")
        f.write("- **Random State:** 42 (for reproducibility)\n")
        f.write("- **Models Tested:** Logistic Regression, Linear SVM, Random Forest, Multinomial Naive Bayes\n")
    
    print(f"[INFO] Generated comprehensive report: {report_path}")
    return report_path

# =========================
# MAIN
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive hyperparameter experiments for all models in IndoHoaxDetector."
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
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for results (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--skip_plots",
        action='store_true',
        help="Skip generating visualization plots"
    )
    
    args = parser.parse_args()
    
    print(f"[INFO] Starting comprehensive model comparison experiments")
    print(f"[INFO] TF-IDF parameters: max_features={args.max_features}, ngram_range=({args.ngram_min},{args.ngram_max})")
    print(f"[INFO] Output directory: {args.output_dir}")
    print(f"[INFO] Models to test: {list(EXPERIMENT_CONFIGS.keys())}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run all experiments
    try:
        results = run_all_experiments(
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
            create_comprehensive_visualizations(results_df, args.output_dir)
        
        # Generate comprehensive report
        report_path = generate_comprehensive_report(
            results_df,
            args.output_dir,
            args.max_features,
            args.ngram_min,
            args.ngram_max
        )
        
        print(f"\n{'='*80}")
        print("ALL MODEL EXPERIMENTS COMPLETED")
        print(f"{'='*80}")
        print(f"Results saved to: {args.output_dir}")
        print(f"Detailed CSV: {csv_path}")
        print(f"Comprehensive Report: {report_path}")
        
        # Print overall best result summary
        successful_df = results_df[results_df['success']]
        if not successful_df.empty:
            best_result = successful_df.loc[successful_df['f1_mean'].idxmax()]
            print(f"\nOverall Best Model: {best_result['model_name']}")
            print(f"Best Hyperparameter: {best_result['param_name']} = {best_result['param_value']}")
            print(f"Best F1 Score: {best_result['f1_mean']:.4f} ± {best_result['f1_std']:.4f}")
        
    except Exception as e:
        print(f"[ERROR] Experiments failed: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()