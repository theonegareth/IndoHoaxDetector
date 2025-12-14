#!/usr/bin/env python3
"""
Regenerate comprehensive experiment summary CSV from JSON metrics files.
This fixes missing metrics for logistic regression experiments.
"""

import json
import os
import pandas as pd
import numpy as np
from pathlib import Path

def parse_experiment_id(exp_id):
    """Parse experiment ID to extract model, param_value, max_features, ngram_range."""
    # Example: logreg_0.01_mf1000_ng1-1
    parts = exp_id.split('_')
    if len(parts) != 4:
        return None
    model = parts[0]
    param_value = float(parts[1])
    mf_part = parts[2]
    ng_part = parts[3]
    # mf1000 -> 1000
    max_features = int(mf_part[2:])
    # ng1-1 -> (1,1)
    ng_range = ng_part[2:].split('-')
    ngram_min = int(ng_range[0])
    ngram_max = int(ng_range[1])
    return model, param_value, max_features, ngram_min, ngram_max

def find_metrics_file(model, param_value, max_features, ngram_min, ngram_max, results_dir):
    """Determine the metrics JSON file name based on model and parameters."""
    if model == "logreg":
        return f"metrics_c{param_value}_mf{max_features}_ng{ngram_min}-{ngram_max}.json"
    elif model == "svm":
        return f"svm_metrics_c{param_value}_mf{max_features}_ng{ngram_min}-{ngram_max}.json"
    elif model == "nb":
        return f"nb_metrics_a{param_value}_mf{max_features}_ng{ngram_min}-{ngram_max}.json"
    elif model == "rf":
        return f"rf_metrics_n{param_value}_mf{max_features}_ng{ngram_min}-{ngram_max}.json"
    else:
        return None

def load_metrics(json_path):
    """Load metrics from a JSON file."""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def main():
    results_dir = Path("comprehensive_results")
    config_path = results_dir / "experiment_configuration.json"
    if not config_path.exists():
        print("Error: experiment_configuration.json not found.")
        return

    with open(config_path, 'r') as f:
        config = json.load(f)

    experiments = config['experiments']
    rows = []

    for exp in experiments:
        exp_id = exp['experiment_id']
        model = exp['model']
        param_value = exp['param_value']
        max_features = exp['max_features']
        ngram_min = exp['ngram_min']
        ngram_max = exp['ngram_max']

        # Determine metrics file
        metrics_file = find_metrics_file(model, param_value, max_features, ngram_min, ngram_max, results_dir)
        metrics_path = results_dir / metrics_file if metrics_file else None

        if metrics_path and metrics_path.exists():
            metrics = load_metrics(metrics_path)
            if metrics:
                success = True
                accuracy_mean = metrics.get('validation_accuracy_mean', np.nan)
                accuracy_std = metrics.get('validation_accuracy_std', np.nan)
                precision_mean = metrics.get('validation_precision_mean', np.nan)
                precision_std = metrics.get('validation_precision_std', np.nan)
                recall_mean = metrics.get('validation_recall_mean', np.nan)
                recall_std = metrics.get('validation_recall_std', np.nan)
                f1_mean = metrics.get('validation_f1_mean', np.nan)
                f1_std = metrics.get('validation_f1_std', np.nan)
                training_duration = metrics.get('training_duration', np.nan)
                cv_folds = metrics.get('cv_folds', np.nan)
                n_samples = metrics.get('n_samples', np.nan)
                timestamp = metrics.get('timestamp', '')
            else:
                success = False
                accuracy_mean = accuracy_std = precision_mean = precision_std = recall_mean = recall_std = f1_mean = f1_std = training_duration = cv_folds = n_samples = np.nan
                timestamp = ''
        else:
            success = False
            accuracy_mean = accuracy_std = precision_mean = precision_std = recall_mean = recall_std = f1_mean = f1_std = training_duration = cv_folds = n_samples = np.nan
            timestamp = ''

        # Experiment duration is not stored in JSON; we'll keep NaN for now.
        row = {
            'experiment_id': exp_id,
            'model': model,
            'param_name': exp['param_name'],
            'param_value': param_value,
            'max_features': max_features,
            'ngram_min': ngram_min,
            'ngram_max': ngram_max,
            'success': success,
            'experiment_duration': np.nan,  # not available in JSON
            'timestamp': timestamp,
            'accuracy_mean': accuracy_mean,
            'accuracy_std': accuracy_std,
            'precision_mean': precision_mean,
            'precision_std': precision_std,
            'recall_mean': recall_mean,
            'recall_std': recall_std,
            'f1_mean': f1_mean,
            'f1_std': f1_std,
            'training_duration': training_duration,
            'cv_folds': cv_folds,
            'n_samples': n_samples
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    output_path = results_dir / "comprehensive_experiment_results_fixed.csv"
    df.to_csv(output_path, index=False)
    print(f"Generated fixed results CSV at {output_path}")

    # Generate summary (successful only)
    successful_df = df[df['success']].copy()
    if not successful_df.empty:
        summary_df = successful_df.sort_values('f1_mean', ascending=False)
        summary_path = results_dir / "comprehensive_experiment_summary_fixed.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Generated fixed summary CSV at {summary_path}")

        # Best configurations per model
        best_configs = []
        for model in successful_df['model'].unique():
            model_df = successful_df[successful_df['model'] == model]
            if not model_df.empty:
                best = model_df.loc[model_df['f1_mean'].idxmax()]
                best_configs.append(best)
        if best_configs:
            best_df = pd.DataFrame(best_configs)
            best_path = results_dir / "best_configurations_fixed.csv"
            best_df.to_csv(best_path, index=False)
            print(f"Generated best configurations CSV at {best_path}")

    # Also generate visualizations
    try:
        from run_comprehensive_experiments import create_comprehensive_visualizations
        create_comprehensive_visualizations(df, str(results_dir))
        print("Generated visualizations.")
    except Exception as e:
        print(f"Visualization generation failed: {e}")

    print("Done.")

if __name__ == "__main__":
    main()