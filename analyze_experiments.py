#!/usr/bin/env python3
"""
Deep analysis of comprehensive experiment results.
Generates additional visualizations and insights.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data():
    """Load merged experiment results."""
    df = pd.read_csv('comprehensive_results/comprehensive_experiment_summary_merged.csv')
    # Ensure numeric columns
    numeric_cols = ['accuracy_mean', 'accuracy_std', 'precision_mean', 'precision_std',
                    'recall_mean', 'recall_std', 'f1_mean', 'f1_std', 'training_duration']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def plot_model_comparison(df, output_dir='comprehensive_results'):
    """Bar chart comparing average F1 score per model."""
    plt.figure(figsize=(10, 6))
    model_avg = df.groupby('model')['f1_mean'].agg(['mean', 'std']).reset_index()
    model_avg = model_avg.sort_values('mean', ascending=False)
    bars = plt.bar(model_avg['model'], model_avg['mean'], yerr=model_avg['std'], capsize=5, alpha=0.7)
    plt.title('Model Performance Comparison (Average F1 Score)', fontsize=14, fontweight='bold')
    plt.xlabel('Model')
    plt.ylabel('F1 Score')
    plt.ylim(0, 1)
    for bar, mean_val in zip(bars, model_avg['mean']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300)
    plt.close()

def plot_parameter_sensitivity(df, output_dir='comprehensive_results'):
    """Line plots of F1 vs parameter value for each model."""
    models = df['model'].unique()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for idx, model in enumerate(models[:4]):  # up to 4 models
        ax = axes[idx]
        model_df = df[df['model'] == model]
        # Group by parameter value (C, alpha, n_estimators)
        param_name = model_df['param_name'].iloc[0]
        param_perf = model_df.groupby('param_value')['f1_mean'].agg(['mean', 'std']).reset_index()
        ax.errorbar(param_perf['param_value'], param_perf['mean'], yerr=param_perf['std'],
                   marker='o', capsize=5, linewidth=2)
        ax.set_title(f'{model.upper()} Parameter Sensitivity ({param_name})', fontsize=12, fontweight='bold')
        ax.set_xlabel(param_name)
        ax.set_ylabel('F1 Score')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameter_sensitivity.png'), dpi=300)
    plt.close()

def plot_tfidf_impact(df, output_dir='comprehensive_results'):
    """Heatmap of F1 score across max_features and ngram_range."""
    # Create a pivot table for average F1
    pivot = df.pivot_table(values='f1_mean', index='max_features',
                           columns=['ngram_min', 'ngram_max'], aggfunc='mean')
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='viridis', cbar_kws={'label': 'F1 Score'})
    plt.title('TF‑IDF Parameter Impact on F1 Score (Average across models)', fontsize=14, fontweight='bold')
    plt.xlabel('N‑gram Range')
    plt.ylabel('Max Features')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tfidf_impact_heatmap.png'), dpi=300)
    plt.close()

def plot_training_time_vs_performance(df, output_dir='comprehensive_results'):
    """Scatter plot of training duration vs F1 score."""
    plt.figure(figsize=(10, 6))
    colors = {'svm': 'red', 'logreg': 'blue', 'rf': 'green', 'nb': 'orange'}
    for model, color in colors.items():
        subset = df[df['model'] == model]
        if len(subset) > 0:
            plt.scatter(subset['training_duration'], subset['f1_mean'],
                       label=model.upper(), alpha=0.7, c=color, s=80)
    plt.xlabel('Training Duration (seconds)')
    plt.ylabel('F1 Score')
    plt.title('Training Time vs Performance', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_vs_performance.png'), dpi=300)
    plt.close()

def generate_analysis_report(df, output_dir='comprehensive_results'):
    """Generate a text analysis report."""
    report_path = os.path.join(output_dir, 'detailed_analysis.txt')
    with open(report_path, 'w') as f:
        f.write("=== Detailed Experiment Analysis ===\n\n")
        f.write(f"Total experiments: {len(df)}\n")
        f.write(f"Successful experiments: {df['success'].sum()}\n")
        f.write(f"Models: {', '.join(df['model'].unique())}\n\n")

        # Best overall
        best = df.loc[df['f1_mean'].idxmax()]
        f.write("BEST OVERALL CONFIGURATION:\n")
        f.write(f"  Model: {best['model'].upper()}\n")
        f.write(f"  Parameter: {best['param_name']} = {best['param_value']}\n")
        f.write(f"  TF‑IDF: max_features={best['max_features']}, ngram_range=({best['ngram_min']},{best['ngram_max']})\n")
        f.write(f"  F1: {best['f1_mean']:.4f} ± {best['f1_std']:.4f}\n")
        f.write(f"  Accuracy: {best['accuracy_mean']:.4f} ± {best['accuracy_std']:.4f}\n")
        f.write(f"  Training time: {best['training_duration']:.2f} seconds\n\n")

        # Best per model
        f.write("BEST CONFIGURATION PER MODEL:\n")
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            if not model_df.empty:
                best_model = model_df.loc[model_df['f1_mean'].idxmax()]
                f.write(f"  {model.upper()}:\n")
                f.write(f"    F1: {best_model['f1_mean']:.4f} ± {best_model['f1_std']:.4f}\n")
                f.write(f"    Params: {best_model['param_name']}={best_model['param_value']}, "
                       f"max_features={best_model['max_features']}, ngram=({best_model['ngram_min']},{best_model['ngram_max']})\n")
                f.write(f"    Training time: {best_model['training_duration']:.2f}s\n\n")

        # Parameter insights
        f.write("PARAMETER INSIGHTS:\n")
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            if len(model_df) > 0:
                param_name = model_df['param_name'].iloc[0]
                best_param = model_df.loc[model_df['f1_mean'].idxmax(), 'param_value']
                f.write(f"  {model.upper()} best {param_name}: {best_param}\n")
        f.write("\n")

        # TF‑IDF insights
        f.write("TF‑IDF INSIGHTS:\n")
        tfidf_best = df.loc[df['f1_mean'].idxmax()]
        f.write(f"  Best max_features: {tfidf_best['max_features']}\n")
        f.write(f"  Best ngram_range: ({tfidf_best['ngram_min']},{tfidf_best['ngram_max']})\n")
        # Average performance per max_features
        for mf in sorted(df['max_features'].unique()):
            avg_f1 = df[df['max_features'] == mf]['f1_mean'].mean()
            f.write(f"  max_features={mf}: average F1 = {avg_f1:.4f}\n")
        f.write("\n")

        # Time‑performance trade‑off
        f.write("TIME‑PERFORMANCE TRADE‑OFF:\n")
        fastest = df.loc[df['training_duration'].idxmin()]
        f.write(f"  Fastest experiment: {fastest['model']} ({fastest['training_duration']:.2f}s) F1={fastest['f1_mean']:.4f}\n")
        slowest = df.loc[df['training_duration'].idxmax()]
        f.write(f"  Slowest experiment: {slowest['model']} ({slowest['training_duration']:.2f}s) F1={slowest['f1_mean']:.4f}\n")
        f.write(f"  Correlation (training time vs F1): {df['training_duration'].corr(df['f1_mean']):.3f}\n")

    print(f"Detailed analysis written to {report_path}")

def main():
    output_dir = 'comprehensive_results'
    os.makedirs(output_dir, exist_ok=True)

    df = load_data()
    print(f"Loaded {len(df)} experiments.")

    # Generate plots
    plot_model_comparison(df, output_dir)
    plot_parameter_sensitivity(df, output_dir)
    plot_tfidf_impact(df, output_dir)
    plot_training_time_vs_performance(df, output_dir)

    # Generate analysis report
    generate_analysis_report(df, output_dir)

    print("Analysis complete. Plots and report saved in", output_dir)

if __name__ == '__main__':
    main()