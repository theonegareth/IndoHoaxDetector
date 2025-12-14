#!/usr/bin/env python3
"""
Comprehensive Experiment Framework for IndoHoaxDetector.

This script runs systematic experiments across multiple models and TF-IDF parameter
combinations to find optimal hyperparameters for hoax detection.

Usage:
    python run_comprehensive_experiments.py
    python run_comprehensive_experiments.py --models logreg svm
    python run_comprehensive_experiments.py --max_features 5000 --ngram_range 1-2
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple
from itertools import product

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# CONFIGURATION
# =========================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "comprehensive_results")

# Experiment parameter grids
TFIDF_MAX_FEATURES = [1000, 3000, 5000, 10000]
TFIDF_NGRAM_RANGES = [(1, 1), (1, 2), (1, 3)]

MODEL_CONFIGS = {
    "logreg": {
        "script": os.path.join(SCRIPT_DIR, "train_logreg.py"),
        "param_name": "c_value",
        "param_values": [0.01, 0.1, 1.0, 10.0, 100.0],
        "param_flag": "--c_value"
    },
    "svm": {
        "script": os.path.join(SCRIPT_DIR, "train_svm.py"),
        "param_name": "c_value", 
        "param_values": [0.01, 0.1, 1.0, 10.0, 100.0],
        "param_flag": "--c_value"
    },
    "nb": {
        "script": os.path.join(SCRIPT_DIR, "train_nb.py"),
        "param_name": "alpha",
        "param_values": [0.1, 0.5, 1.0, 2.0, 5.0],
        "param_flag": "--alpha"
    },
    "rf": {
        "script": os.path.join(SCRIPT_DIR, "train_rf.py"),
        "param_name": "n_estimators",
        "param_values": [50, 100, 200, 500],
        "param_flag": "--n_estimators"
    }
}

# =========================
# EXPERIMENT EXECUTION
# =========================

def create_experiment_matrix(models: List[str]) -> List[Dict[str, Any]]:
    """Create comprehensive experiment matrix with all parameter combinations."""
    
    experiments = []
    
    for model in models:
        if model not in MODEL_CONFIGS:
            print(f"[WARNING] Unknown model: {model}. Skipping.")
            continue
            
        config = MODEL_CONFIGS[model]
        
        # Create all combinations of TF-IDF parameters and model parameters
        for max_features in TFIDF_MAX_FEATURES:
            for ngram_range in TFIDF_NGRAM_RANGES:
                for param_value in config["param_values"]:
                    experiment = {
                        "model": model,
                        "script": config["script"],
                        "param_name": config["param_name"],
                        "param_value": param_value,
                        "param_flag": config["param_flag"],
                        "max_features": max_features,
                        "ngram_min": ngram_range[0],
                        "ngram_max": ngram_range[1],
                        "experiment_id": f"{model}_{param_value}_mf{max_features}_ng{ngram_range[0]}-{ngram_range[1]}"
                    }
                    experiments.append(experiment)
    
    return experiments

def run_single_experiment(experiment: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """Execute a single experiment with specified parameters."""
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {experiment['experiment_id']}")
    print(f"Model: {experiment['model']}, {experiment['param_name']}={experiment['param_value']}")
    print(f"TF-IDF: max_features={experiment['max_features']}, ngram_range=({experiment['ngram_min']},{experiment['ngram_max']})")
    print(f"{'='*80}")
    
    # Prepare command
    cmd = [
        sys.executable,
        experiment["script"],
        experiment["param_flag"], str(experiment["param_value"]),
        "--max_features", str(experiment["max_features"]),
        "--ngram_min", str(experiment["ngram_min"]),
        "--ngram_max", str(experiment["ngram_max"]),
        "--output_dir", output_dir
    ]
    
    # Run the training script
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=3600  # 1 hour timeout per experiment
        )
        success = True
        stdout = result.stdout
        stderr = result.stderr
    except subprocess.CalledProcessError as e:
        success = False
        stdout = e.stdout
        stderr = e.stderr
        print(f"[ERROR] Experiment failed: {e}")
    except subprocess.TimeoutExpired:
        success = False
        stdout = ""
        stderr = "Experiment timed out after 1 hour"
        print(f"[ERROR] Experiment timed out: {experiment['experiment_id']}")
    
    experiment_duration = time.time() - start_time
    
    # Try to load metrics if successful
    metrics = None
    if success:
        # Determine metrics file pattern based on model
        model = experiment["model"]
        param_value = experiment["param_value"]
        max_features = experiment["max_features"]
        ngram_min = experiment["ngram_min"]
        ngram_max = experiment["ngram_max"]
        
        if model == "logreg":
            metrics_file = f"metrics_c{param_value}_mf{max_features}_ng{ngram_min}-{ngram_max}.json"
        elif model == "svm":
            metrics_file = f"svm_metrics_c{param_value}_mf{max_features}_ng{ngram_min}-{ngram_max}.json"
        elif model == "nb":
            metrics_file = f"nb_metrics_a{param_value}_mf{max_features}_ng{ngram_min}-{ngram_max}.json"
        elif model == "rf":
            metrics_file = f"rf_metrics_n{param_value}_mf{max_features}_ng{ngram_min}-{ngram_max}.json"
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
        **experiment,
        'success': success,
        'experiment_duration': experiment_duration,
        'timestamp': datetime.now().isoformat(),
        'stdout': stdout,
        'stderr': stderr,
        'metrics': metrics
    }

def run_experiments(experiments: List[Dict[str, Any]], output_dir: str, max_parallel: int = 2) -> List[Dict[str, Any]]:
    """Run all experiments with optional parallel execution."""
    
    print(f"[INFO] Starting {len(experiments)} experiments")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Max parallel experiments: {max_parallel}")
    
    results = []
    completed = 0
    
    for experiment in experiments:
        result = run_single_experiment(experiment, output_dir)
        results.append(result)
        completed += 1
        
        # Print progress
        if result['success'] and result['metrics']:
            m = result['metrics']
            print(f"[PROGRESS] {completed}/{len(experiments)} - {experiment['experiment_id']}: "
                  f"Acc={m.get('validation_accuracy_mean', 0):.4f}±{m.get('validation_accuracy_std', 0):.4f}, "
                  f"F1={m.get('validation_f1_mean', 0):.4f}±{m.get('validation_f1_std', 0):.4f}")
        else:
            print(f"[PROGRESS] {completed}/{len(experiments)} - {experiment['experiment_id']}: FAILED")
    
    return results

# =========================
# RESULTS PROCESSING
# =========================

def process_results(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Process experiment results into a comprehensive DataFrame."""
    
    processed_data = []
    
    for result in results:
        if not result['success'] or not result['metrics']:
            # Include failed experiments with NaN values
            processed_data.append({
                'experiment_id': result['experiment_id'],
                'model': result['model'],
                'param_name': result['param_name'],
                'param_value': result['param_value'],
                'max_features': result['max_features'],
                'ngram_min': result['ngram_min'],
                'ngram_max': result['ngram_max'],
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
            'experiment_id': result['experiment_id'],
            'model': result['model'],
            'param_name': result['param_name'],
            'param_value': result['param_value'],
            'max_features': result['max_features'],
            'ngram_min': result['ngram_min'],
            'ngram_max': result['ngram_max'],
            'success': True,
            'experiment_duration': result['experiment_duration'],
            'timestamp': result['timestamp'],
            'accuracy_mean': m.get('validation_accuracy_mean', float('nan')),
            'accuracy_std': m.get('validation_accuracy_std', float('nan')),
            'precision_mean': m.get('validation_precision_mean', float('nan')),
            'precision_std': m.get('validation_precision_std', float('nan')),
            'recall_mean': m.get('validation_recall_mean', float('nan')),
            'recall_std': m.get('validation_recall_std', float('nan')),
            'f1_mean': m.get('validation_f1_mean', float('nan')),
            'f1_std': m.get('validation_f1_std', float('nan')),
            'training_duration': m.get('training_duration', float('nan')),
            'cv_folds': m.get('cv_folds', float('nan')),
            'n_samples': m.get('n_samples', float('nan'))
        })
    
    return pd.DataFrame(processed_data)

def save_results(results_df: pd.DataFrame, output_dir: str):
    """Save comprehensive results to multiple formats."""
    
    # Save detailed results
    csv_path = os.path.join(output_dir, "comprehensive_experiment_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"[INFO] Saved detailed results to: {csv_path}")
    
    # Save summary (successful experiments only)
    successful_df = results_df[results_df['success']].copy()
    if not successful_df.empty:
        # Sort by F1 score for summary
        summary_df = successful_df.sort_values('f1_mean', ascending=False)
        summary_path = os.path.join(output_dir, "comprehensive_experiment_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"[INFO] Saved summary to: {summary_path}")
        
        # Save best configurations
        best_configs = []
        for model in successful_df['model'].unique():
            model_df = successful_df[successful_df['model'] == model]
            if not model_df.empty:
                best_config = model_df.loc[model_df['f1_mean'].idxmax()]
                best_configs.append(best_config)
        
        if best_configs:
            best_df = pd.DataFrame(best_configs)
            best_path = os.path.join(output_dir, "best_configurations.csv")
            best_df.to_csv(best_path, index=False)
            print(f"[INFO] Saved best configurations to: {best_path}")
    
    return csv_path

# =========================
# VISUALIZATION
# =========================

def create_comprehensive_visualizations(results_df: pd.DataFrame, output_dir: str):
    """Create comprehensive visualizations of all experimental results."""
    
    successful_df = results_df[results_df['success']].copy()
    if successful_df.empty:
        print("[WARNING] No successful experiments to visualize.")
        return
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 10)
    
    # Create comprehensive figure with multiple subplots
    fig = plt.figure(figsize=(20, 24))
    
    # 1. Model Performance Comparison
    ax1 = plt.subplot(4, 2, 1)
    model_performance = successful_df.groupby('model')['f1_mean'].agg(['mean', 'std']).reset_index()
    bars = ax1.bar(model_performance['model'], model_performance['mean'], 
                   yerr=model_performance['std'], capsize=5, alpha=0.7)
    ax1.set_title('Model Performance Comparison (F1 Score)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('F1 Score')
    ax1.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, mean_val in zip(bars, model_performance['mean']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. TF-IDF max_features impact
    ax2 = plt.subplot(4, 2, 2)
    feature_performance = successful_df.groupby('max_features')['f1_mean'].agg(['mean', 'std']).reset_index()
    ax2.errorbar(feature_performance['max_features'], feature_performance['mean'],
                yerr=feature_performance['std'], marker='o', capsize=5, linewidth=2)
    ax2.set_title('Impact of TF-IDF max_features on Performance', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Max Features')
    ax2.set_ylabel('F1 Score')
    ax2.set_xscale('log')
    
    # 3. N-gram range impact
    ax3 = plt.subplot(4, 2, 3)
    ngram_performance = successful_df.groupby(['ngram_min', 'ngram_max'])['f1_mean'].agg(['mean', 'std']).reset_index()
    ngram_labels = [f"({row['ngram_min']},{row['ngram_max']})" for _, row in ngram_performance.iterrows()]
    bars = ax3.bar(ngram_labels, ngram_performance['mean'], 
                   yerr=ngram_performance['std'], capsize=5, alpha=0.7)
    ax3.set_title('Impact of N-gram Range on Performance', fontsize=14, fontweight='bold')
    ax3.set_ylabel('F1 Score')
    ax3.set_xticklabels(ngram_labels, rotation=45)
    
    # 4. Heatmap: Model vs TF-IDF parameters
    ax4 = plt.subplot(4, 2, 4)
    pivot_data = successful_df.pivot_table(values='f1_mean', 
                                          index='model', 
                                          columns=['max_features', 'ngram_min', 'ngram_max'],
                                          aggfunc='mean')
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis', ax=ax4)
    ax4.set_title('Performance Heatmap: Model vs TF-IDF Parameters', fontsize=14, fontweight='bold')
    
    # 5. Parameter sensitivity analysis for each model
    models = successful_df['model'].unique()
    for i, model in enumerate(models[:4]):  # Show top 4 models
        ax = plt.subplot(4, 2, 5 + i)
        model_df = successful_df[successful_df['model'] == model]
        
        # Group by parameter value and calculate mean performance
        param_performance = model_df.groupby('param_value')['f1_mean'].agg(['mean', 'std']).reset_index()
        ax.errorbar(param_performance['param_value'], param_performance['mean'],
                   yerr=param_performance['std'], marker='o', capsize=5, linewidth=2)
        ax.set_title(f'{model.upper()} Parameter Sensitivity', fontsize=12, fontweight='bold')
        ax.set_xlabel('Parameter Value')
        ax.set_ylabel('F1 Score')
        ax.set_xscale('log')
    
    plt.tight_layout()
    
    # Save comprehensive plot
    plot_path = os.path.join(output_dir, "comprehensive_experiment_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Saved comprehensive visualization to: {plot_path}")
    
    plt.close()

# =========================
# REPORTING
# =========================

def generate_comprehensive_report(results_df: pd.DataFrame, output_dir: str):
    """Generate a comprehensive experiment report with detailed analysis."""
    
    successful_df = results_df[results_df['success']].copy()
    
    report_path = os.path.join(output_dir, "comprehensive_experiment_report.md")
    
    with open(report_path, 'w') as f:
        f.write("# Comprehensive Experiment Report: IndoHoaxDetector\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write(f"- **Total Experiments:** {len(results_df)}\n")
        f.write(f"- **Successful Experiments:** {len(successful_df)}\n")
        f.write(f"- **Failed Experiments:** {len(results_df) - len(successful_df)}\n")
        f.write(f"- **Success Rate:** {len(successful_df)/len(results_df)*100:.1f}%\n")
        
        if not successful_df.empty:
            f.write(f"- **Dataset Size:** {int(successful_df['n_samples'].iloc[0])} samples\n")
            f.write(f"- **Cross-validation Folds:** {int(successful_df['cv_folds'].iloc[0])}\n")
        
        f.write("\n## Best Performing Configurations\n\n")
        
        if successful_df.empty:
            f.write("❌ **All experiments failed.** Check the logs for details.\n\n")
        else:
            # Overall best
            best_overall = successful_df.loc[successful_df['f1_mean'].idxmax()]
            f.write("### Overall Best Configuration\n\n")
            f.write(f"- **Model:** {best_overall['model'].upper()}\n")
            f.write(f"- **Parameter:** {best_overall['param_name']} = {best_overall['param_value']}\n")
            f.write(f"- **TF-IDF:** max_features={best_overall['max_features']}, ngram_range=({best_overall['ngram_min']},{best_overall['ngram_max']})\n")
            f.write(f"- **F1 Score:** {best_overall['f1_mean']:.4f} ± {best_overall['f1_std']:.4f}\n")
            f.write(f"- **Accuracy:** {best_overall['accuracy_mean']:.4f} ± {best_overall['accuracy_std']:.4f}\n")
            f.write(f"- **Precision:** {best_overall['precision_mean']:.4f} ± {best_overall['precision_std']:.4f}\n")
            f.write(f"- **Recall:** {best_overall['recall_mean']:.4f} ± {best_overall['recall_std']:.4f}\n\n")
            
            # Best by model
            f.write("### Best Configuration by Model\n\n")
            for model in successful_df['model'].unique():
                model_df = successful_df[successful_df['model'] == model]
                if not model_df.empty:
                    best_model = model_df.loc[model_df['f1_mean'].idxmax()]
                    f.write(f"#### {model.upper()}\n")
                    f.write(f"- **Parameter:** {best_model['param_name']} = {best_model['param_value']}\n")
                    f.write(f"- **TF-IDF:** max_features={best_model['max_features']}, ngram_range=({best_model['ngram_min']},{best_model['ngram_max']})\n")
                    f.write(f"- **F1 Score:** {best_model['f1_mean']:.4f} ± {best_model['f1_std']:.4f}\n\n")
            
            # Parameter analysis
            f.write("## Parameter Analysis\n\n")
            
            # TF-IDF analysis
            f.write("### TF-IDF Parameter Impact\n\n")
            tfidf_analysis = successful_df.groupby(['max_features', 'ngram_min', 'ngram_max'])['f1_mean'].agg(['mean', 'std', 'count']).reset_index()
            tfidf_analysis = tfidf_analysis.sort_values('mean', ascending=False)
            
            f.write("| Max Features | N-gram Range | Mean F1 | Std F1 | Count |\n")
            f.write("|-------------|--------------|---------|--------|-------|\n")
            for _, row in tfidf_analysis.head(10).iterrows():
                f.write(f"| {row['max_features']} | ({row['ngram_min']},{row['ngram_max']}) | ")
                f.write(f"{row['mean']:.4f} | {row['std']:.4f} | {int(row['count'])} |\n")
            f.write("\n")
            
            # Model comparison
            f.write("### Model Performance Summary\n\n")
            model_summary = successful_df.groupby('model')['f1_mean'].agg(['mean', 'std', 'min', 'max']).reset_index()
            model_summary = model_summary.sort_values('mean', ascending=False)
            
            f.write("| Model | Mean F1 | Std F1 | Min F1 | Max F1 |\n")
            f.write("|-------|---------|--------|--------|--------|\n")
            for _, row in model_summary.iterrows():
                f.write(f"| {row['model'].upper()} | {row['mean']:.4f} | {row['std']:.4f} | ")
                f.write(f"{row['min']:.4f} | {row['max']:.4f} |\n")
            f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("### For Production Deployment\n\n")
            f.write(f"1. **Use {best_overall['model'].upper()}** with {best_overall['param_name']}={best_overall['param_value']}\n")
            f.write(f"2. **TF-IDF Configuration:** max_features={best_overall['max_features']}, ngram_range=({best_overall['ngram_min']},{best_overall['ngram_max']})\n")
            f.write("3. **Expected Performance:**\n")
            f.write(f"   - F1 Score: {best_overall['f1_mean']:.4f} ± {best_overall['f1_std']:.4f}\n")
            f.write(f"   - Accuracy: {best_overall['accuracy_mean']:.4f} ± {best_overall['accuracy_std']:.4f}\n")
            f.write("\n")
            
            f.write("### For Further Optimization\n\n")
            f.write("1. **Fine-tune around best configurations** with smaller parameter grids\n")
            f.write("2. **Consider ensemble methods** combining top-performing models\n")
            f.write("3. **Experiment with advanced feature engineering** techniques\n")
            f.write("4. **Collect more training data** if possible to improve generalization\n\n")
        
        f.write("## Files Generated\n\n")
        f.write("- `comprehensive_experiment_results.csv`: All experimental results\n")
        f.write("- `comprehensive_experiment_summary.csv`: Successful experiments only\n")
        f.write("- `best_configurations.csv`: Best configuration for each model\n")
        f.write("- `comprehensive_experiment_analysis.png`: Comprehensive visualizations\n")
        f.write("- Individual model files: `{model}_model_*.pkl`\n")
        f.write("- Individual vectorizers: `tfidf_vectorizer_*.pkl`\n")
        f.write("- Individual metrics: `{model}_metrics_*.json`\n\n")
        
        f.write("## Technical Details\n\n")
        f.write("- **Models Tested:** Logistic Regression, Linear SVM, Multinomial Naive Bayes, Random Forest\n")
        f.write("- **TF-IDF max_features:** [1000, 3000, 5000, 10000]\n")
        f.write("- **TF-IDF ngram_range:** [(1,1), (1,2), (1,3)]\n")
        f.write("- **Evaluation:** 5-fold stratified cross-validation\n")
        f.write("- **Metrics:** Accuracy, Precision, Recall, F1 Score\n")
        f.write("- **Random State:** 42 (for reproducibility)\n")
    
    print(f"[INFO] Generated comprehensive report: {report_path}")
    return report_path

# =========================
# MAIN
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive experiments for IndoHoaxDetector with multiple models and TF-IDF parameters."
    )
    parser.add_argument(
        "--models",
        nargs='+',
        default=list(MODEL_CONFIGS.keys()),
        choices=list(MODEL_CONFIGS.keys()),
        help=f"Models to test (default: {list(MODEL_CONFIGS.keys())})"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for results (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--max_parallel",
        type=int,
        default=2,
        help="Maximum number of parallel experiments (default: 2)"
    )
    parser.add_argument(
        "--skip_plots",
        action='store_true',
        help="Skip generating visualization plots"
    )
    
    args = parser.parse_args()
    
    print(f"[INFO] Starting comprehensive experiments")
    print(f"[INFO] Models to test: {args.models}")
    print(f"[INFO] Output directory: {args.output_dir}")
    print(f"[INFO] Max parallel experiments: {args.max_parallel}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create experiment matrix
    experiments = create_experiment_matrix(args.models)
    print(f"[INFO] Created {len(experiments)} experiments")
    
    # Save experiment configuration
    config_path = os.path.join(args.output_dir, "experiment_configuration.json")
    with open(config_path, 'w') as f:
        json.dump({
            "models": args.models,
            "experiments": experiments,
            "tfidf_max_features": TFIDF_MAX_FEATURES,
            "tfidf_ngram_ranges": TFIDF_NGRAM_RANGES,
            "model_configs": MODEL_CONFIGS
        }, f, indent=2)
    print(f"[INFO] Saved experiment configuration to: {config_path}")
    
    # Run experiments
    try:
        results = run_experiments(experiments, args.output_dir, args.max_parallel)
        
        # Process and save results
        results_df = process_results(results)
        csv_path = save_results(results_df, args.output_dir)
        
        # Generate visualizations
        if not args.skip_plots:
            create_comprehensive_visualizations(results_df, args.output_dir)
        
        # Generate comprehensive report
        report_path = generate_comprehensive_report(results_df, args.output_dir)
        
        print(f"\n{'='*80}")
        print("COMPREHENSIVE EXPERIMENTS COMPLETED")
        print(f"{'='*80}")
        print(f"Results saved to: {args.output_dir}")
        print(f"Detailed CSV: {csv_path}")
        print(f"Comprehensive Report: {report_path}")
        
        # Print best result summary
        successful_df = results_df[results_df['success']]
        if not successful_df.empty:
            best_result = successful_df.loc[successful_df['f1_mean'].idxmax()]
            print(f"\nBest Overall Configuration:")
            print(f"Model: {best_result['model'].upper()}")
            print(f"Parameter: {best_result['param_name']} = {best_result['param_value']}")
            print(f"TF-IDF: max_features={best_result['max_features']}, ngram_range=({best_result['ngram_min']},{best_result['ngram_max']})")
            print(f"Best F1 Score: {best_result['f1_mean']:.4f} ± {best_result['f1_std']:.4f}")
        
    except Exception as e:
        print(f"[ERROR] Comprehensive experiments failed: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()