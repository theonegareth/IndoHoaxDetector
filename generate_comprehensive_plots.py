#!/usr/bin/env python3
"""
Generate comprehensive plots for IndoHoaxDetector experiment results.
Uses the comprehensive_experiment_summary_merged.csv to create various visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 11

# Colors for models
MODEL_COLORS = {
    'svm': '#3498db',
    'rf': '#2ecc71',
    'nb': '#e74c3c',
    'logreg': '#f39c12',
    'indobert': '#9b59b6'
}

def load_data():
    """Load the comprehensive experiment results."""
    csv_path = 'comprehensive_results/comprehensive_experiment_summary_merged.csv'
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Results CSV not found at {csv_path}")
    df = pd.read_csv(csv_path)
    # Ensure numeric columns
    numeric_cols = ['param_value', 'max_features', 'ngram_min', 'ngram_max',
                    'accuracy_mean', 'accuracy_std', 'precision_mean', 'precision_std',
                    'recall_mean', 'recall_std', 'f1_mean', 'f1_std', 'training_duration']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def plot_hyperparameter_performance(df, output_dir='comprehensive_results/plots'):
    """Plot performance vs hyperparameter for each model."""
    os.makedirs(output_dir, exist_ok=True)

    # Separate by model
    models = df['model'].unique()
    for model in models:
        model_df = df[df['model'] == model].copy()
        if model_df.empty:
            continue

        # Determine hyperparameter name
        param_name = model_df['param_name'].iloc[0]  # e.g., 'c_value', 'n_estimators', 'alpha'
        param_label = param_name.replace('_', ' ').title()

        # Create subplots for each metric
        metrics = ['accuracy_mean', 'precision_mean', 'recall_mean', 'f1_mean']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[idx]
            # Group by max_features and ngram_range
            for max_feat in sorted(model_df['max_features'].unique()):
                for ngram_min in sorted(model_df['ngram_min'].unique()):
                    ngram_max = model_df[model_df['ngram_min'] == ngram_min]['ngram_max'].iloc[0]
                    subset = model_df[(model_df['max_features'] == max_feat) &
                                      (model_df['ngram_min'] == ngram_min) &
                                      (model_df['ngram_max'] == ngram_max)].sort_values('param_value')
                    if subset.empty:
                        continue
                    # Plot line with error bars (std)
                    ax.errorbar(subset['param_value'], subset[metric],
                                yerr=subset[metric.replace('_mean', '_std')],
                                label=f'mf={max_feat}, ng=({ngram_min},{ngram_max})',
                                marker='o', capsize=4, linewidth=2, markersize=6)
            ax.set_xlabel(param_label)
            ax.set_ylabel(label)
            ax.set_title(f'{model.upper()} - {label} vs {param_label}')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.suptitle(f'{model.upper()} Performance Across Hyperparameters', fontsize=18, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model}_hyperparameter_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved hyperparameter performance plot for {model}")

def plot_model_comparison_bar(df, output_dir='comprehensive_results/plots'):
    """Create bar chart comparing average performance across models."""
    os.makedirs(output_dir, exist_ok=True)

    # Compute average performance per model (across all hyperparameters)
    avg_df = df.groupby('model').agg({
        'accuracy_mean': 'mean',
        'precision_mean': 'mean',
        'recall_mean': 'mean',
        'f1_mean': 'mean',
        'training_duration': 'mean'
    }).reset_index()

    # Sort by F1 score descending
    avg_df = avg_df.sort_values('f1_mean', ascending=False)

    # Bar positions
    x = np.arange(len(avg_df))
    width = 0.2
    metrics = ['accuracy_mean', 'precision_mean', 'recall_mean', 'f1_mean']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1']
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']

    fig, ax = plt.subplots(figsize=(14, 8))
    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        values = avg_df[metric].values
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, values, width, label=label, color=color, alpha=0.8)
        # Add value labels on top
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Model', fontsize=14)
    ax.set_ylabel('Score', fontsize=14)
    ax.set_title('Average Performance Across All Hyperparameter Configurations', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(avg_df['model'].str.upper(), rotation=0)
    ax.legend(loc='upper left', fontsize=12)
    ax.set_ylim(0.85, 1.0)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison_average.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved model comparison bar chart")

def plot_heatmap_tfidf(df, output_dir='comprehensive_results/plots'):
    """Create heatmaps of performance across max_features and ngram_range for each model."""
    os.makedirs(output_dir, exist_ok=True)

    models = df['model'].unique()
    for model in models:
        model_df = df[df['model'] == model]
        if model_df.empty:
            continue

        # Create a pivot table for each metric
        metrics = ['accuracy_mean', 'precision_mean', 'recall_mean', 'f1_mean']
        metric_titles = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

        # Determine best hyperparameter value per configuration (average across param_value?)
        # We'll take the max across param_value for each (max_features, ngram_min, ngram_max)
        # First, group by max_features and ngram_range
        model_df['ngram_range'] = model_df.apply(lambda row: f"({row['ngram_min']},{row['ngram_max']})", axis=1)
        grouped = model_df.groupby(['max_features', 'ngram_range']).agg({
            'accuracy_mean': 'max',
            'precision_mean': 'max',
            'recall_mean': 'max',
            'f1_mean': 'max'
        }).reset_index()

        # Pivot for heatmap
        for metric, title in zip(metrics, metric_titles):
            pivot = grouped.pivot(index='max_features', columns='ngram_range', values=metric)
            if pivot.empty:
                continue

            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax, cbar_kws={'label': title})
            ax.set_title(f'{model.upper()} - {title} by TF‑IDF Parameters', fontsize=16)
            ax.set_xlabel('N‑gram Range')
            ax.set_ylabel('Max Features')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{model}_heatmap_{metric}.png'), dpi=300, bbox_inches='tight')
            plt.close()
        print(f"Saved heatmaps for {model}")

def plot_training_time_vs_performance(df, output_dir='comprehensive_results/plots'):
    """Scatter plot of training time vs F1 score."""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    markers = {'svm': 'o', 'rf': 's', 'nb': '^', 'logreg': 'D'}
    for model in df['model'].unique():
        subset = df[df['model'] == model]
        ax.scatter(subset['training_duration'], subset['f1_mean'],
                   label=model.upper(), alpha=0.7, s=80, marker=markers.get(model, 'o'),
                   edgecolors='black', linewidth=0.5)
        # Annotate top 3 points per model
        top3 = subset.nlargest(3, 'f1_mean')
        for _, row in top3.iterrows():
            ax.annotate(f"{row['param_value']}", (row['training_duration'], row['f1_mean']),
                        fontsize=8, alpha=0.8)

    ax.set_xlabel('Training Duration (seconds)', fontsize=14)
    ax.set_ylabel('F1 Score (mean)', fontsize=14)
    ax.set_title('Training Time vs Performance (F1 Score)', fontsize=16)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_time_vs_f1.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved training time vs performance scatter plot")

def plot_f1_distribution_box(df, output_dir='comprehensive_results/plots'):
    """Box plot of F1 distribution across hyperparameters for each model."""
    os.makedirs(output_dir, exist_ok=True)

    # We don't have per‑fold data, but we can show distribution across hyperparameter configurations
    # Use violin plot or box plot of f1_mean per model
    fig, ax = plt.subplots(figsize=(12, 8))
    data = [df[df['model'] == model]['f1_mean'].values for model in df['model'].unique()]
    labels = [model.upper() for model in df['model'].unique()]
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True,
                    meanprops={'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black'})
    # Color boxes
    colors = [MODEL_COLORS.get(model, '#95a5a6') for model in df['model'].unique()]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('F1 Score (mean)', fontsize=14)
    ax.set_title('Distribution of F1 Scores Across Hyperparameter Configurations', fontsize=16)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'f1_distribution_box.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved F1 distribution box plot")

def plot_statistical_significance(output_dir='comprehensive_results/plots'):
    """Visualize statistical significance results if available."""
    # Check if statistical significance CSV exists
    sig_path = 'comprehensive_results/statistical_significance.csv'
    if not os.path.exists(sig_path):
        print("Statistical significance file not found, skipping.")
        return

    df_sig = pd.read_csv(sig_path)
    # The CSV has columns: model_a, model_b, f1_a, f1_b, difference, t_statistic, df, p_value, significant_0.05
    # Rename for clarity
    df_sig = df_sig.rename(columns={'model_a': 'model1', 'model_b': 'model2'})
    # Example: heatmap of p-values
    # Pivot to matrix
    models = sorted(set(df_sig['model1'].unique()) | set(df_sig['model2'].unique()))
    p_matrix = pd.DataFrame(index=models, columns=models, dtype=float)
    for _, row in df_sig.iterrows():
        m1, m2, p = row['model1'], row['model2'], row['p_value']
        p_matrix.loc[m1, m2] = p
        p_matrix.loc[m2, m1] = p
    np.fill_diagonal(p_matrix.values, 1.0)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(p_matrix, annot=True, fmt='.3f', cmap='RdBu_r', center=0.05,
                cbar_kws={'label': 'p-value'}, ax=ax)
    ax.set_title('Pairwise Statistical Significance (p‑values)', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'statistical_significance_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved statistical significance heatmap")

def plot_best_configurations(df, output_dir='comprehensive_results/plots'):
    """Bar chart of best configuration per model."""
    os.makedirs(output_dir, exist_ok=True)

    best_configs = []
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        idx = model_df['f1_mean'].idxmax()
        best = model_df.loc[idx]
        best_configs.append({
            'model': model.upper(),
            'accuracy': best['accuracy_mean'],
            'precision': best['precision_mean'],
            'recall': best['recall_mean'],
            'f1': best['f1_mean'],
            'param': f"{best['param_name']}={best['param_value']}",
            'max_features': best['max_features'],
            'ngram_range': f"({best['ngram_min']},{best['ngram_max']})"
        })
    best_df = pd.DataFrame(best_configs)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(best_df))
    width = 0.2
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        values = best_df[metric].values
        ax.bar(x + (i-1.5)*width, values, width, label=metric.capitalize(), color=color, alpha=0.8)
        # Add value labels
        for j, val in enumerate(values):
            ax.text(x[j] + (i-1.5)*width, val + 0.002, f'{val:.3f}',
                    ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Model', fontsize=14)
    ax.set_ylabel('Score', fontsize=14)
    ax.set_title('Best Configuration Performance per Model', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(best_df['model'])
    ax.legend(loc='upper left')
    ax.set_ylim(0.85, 1.0)
    ax.grid(True, axis='y', alpha=0.3)

    # Add configuration text below
    for i, row in best_df.iterrows():
        config_text = f"{row['param']}, mf={row['max_features']}, ng={row['ngram_range']}"
        ax.text(i, 0.83, config_text, ha='center', va='top', fontsize=9, style='italic')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'best_configurations.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved best configurations bar chart")

def plot_misclassification_analysis(output_dir='comprehensive_results/plots'):
    """Create visualizations for misclassified samples."""
    misclassified_path = 'comprehensive_results/misclassified_samples.csv'
    if not os.path.exists(misclassified_path):
        print("Misclassified samples file not found, skipping.")
        return

    df_mis = pd.read_csv(misclassified_path)
    # Count misclassifications by true label and predicted label
    confusion_counts = df_mis.groupby(['true_class', 'predicted_class']).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(confusion_counts, annot=True, fmt='d', cmap='Blues', ax=ax,
                cbar_kws={'label': 'Count'})
    ax.set_title('Misclassification Matrix (Best SVM Model)', fontsize=16)
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('True Class')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'misclassification_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Bar chart of misclassification counts per true class
    fig, ax = plt.subplots(figsize=(10, 6))
    mis_by_true = df_mis['true_class'].value_counts()
    mis_by_true.plot(kind='bar', color=['#e74c3c', '#3498db'], ax=ax)
    ax.set_title('Misclassified Samples by True Class', fontsize=16)
    ax.set_xlabel('True Class')
    ax.set_ylabel('Count')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    for i, v in enumerate(mis_by_true):
        ax.text(i, v + 0.5, str(v), ha='center', va='bottom', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'misclassification_by_true_class.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved misclassification analysis plots")

def plot_roc_pr_curves(output_dir='comprehensive_results/plots'):
    """Generate ROC and Precision-Recall curves for best models (if probability data available)."""
    # This is a placeholder; we don't have saved probabilities.
    # We could load the best models and compute on test set, but that's heavy.
    # Instead, we'll create a note that ROC/PR curves require prediction probabilities.
    # We'll output a message and skip.
    print("ROC/PR curves require prediction probabilities which are not saved. Skipping.")
    # Optionally, we could create a dummy plot with a message.
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.text(0.5, 0.5, 'ROC/PR curves require prediction probabilities\nnot saved in this experiment run.',
            ha='center', va='center', fontsize=14, transform=ax.transAxes)
    ax.set_title('ROC/PR Curves Not Available', fontsize=16)
    ax.set_xlabel('False Positive Rate / Recall')
    ax.set_ylabel('True Positive Rate / Precision')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_pr_curves_not_available.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_all_in_one_report(output_dir='comprehensive_results/plots'):
    """Create a single figure with multiple subplots summarizing key results."""
    # Load data
    df = load_data()
    # Determine best model overall
    best_idx = df['f1_mean'].idxmax()
    best_row = df.loc[best_idx]
    best_model = best_row['model']
    best_f1 = best_row['f1_mean']
    best_config = f"{best_model.upper()} (C={best_row['param_value']}, mf={best_row['max_features']}, ng=({best_row['ngram_min']},{best_row['ngram_max']}))"

    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Model comparison bar chart (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    avg_df = df.groupby('model').agg({'f1_mean': 'mean'}).reset_index().sort_values('f1_mean', ascending=False)
    colors = [MODEL_COLORS.get(m, '#95a5a6') for m in avg_df['model']]
    ax1.bar(avg_df['model'].str.upper(), avg_df['f1_mean'], color=colors, alpha=0.8)
    ax1.set_title('Average F1 Score by Model', fontsize=14)
    ax1.set_ylabel('F1 Score')
    ax1.set_ylim(0.85, 1.0)
    ax1.tick_params(axis='x', rotation=45)
    for i, v in enumerate(avg_df['f1_mean']):
        ax1.text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

    # 2. Best configuration performance (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    best_configs = []
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        idx = model_df['f1_mean'].idxmax()
        best = model_df.loc[idx]
        best_configs.append({
            'model': model.upper(),
            'f1': best['f1_mean'],
            'accuracy': best['accuracy_mean']
        })
    best_df = pd.DataFrame(best_configs)
    x = np.arange(len(best_df))
    width = 0.35
    ax2.bar(x - width/2, best_df['f1'], width, label='F1', color='#3498db', alpha=0.8)
    ax2.bar(x + width/2, best_df['accuracy'], width, label='Accuracy', color='#2ecc71', alpha=0.8)
    ax2.set_title('Best Configuration per Model', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(best_df['model'], rotation=45)
    ax2.set_ylabel('Score')
    ax2.legend()
    ax2.set_ylim(0.85, 1.0)

    # 3. Training time vs F1 scatter (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    markers = {'svm': 'o', 'rf': 's', 'nb': '^', 'logreg': 'D'}
    for model in df['model'].unique():
        subset = df[df['model'] == model]
        ax3.scatter(subset['training_duration'], subset['f1_mean'],
                   label=model.upper(), alpha=0.7, s=50, marker=markers.get(model, 'o'),
                   edgecolors='black', linewidth=0.5)
    ax3.set_xlabel('Training Duration (s)')
    ax3.set_ylabel('F1 Score')
    ax3.set_title('Training Time vs Performance')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # 4. Heatmap of best model's F1 across TF‑IDF params (middle left)
    ax4 = fig.add_subplot(gs[1, 0])
    best_model_df = df[df['model'] == best_model]
    best_model_df['ngram_range'] = best_model_df.apply(lambda r: f"({r['ngram_min']},{r['ngram_max']})", axis=1)
    pivot = best_model_df.pivot_table(index='max_features', columns='ngram_range', values='f1_mean', aggfunc='max')
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax4, cbar_kws={'label': 'F1'})
    ax4.set_title(f'{best_model.upper()} F1 by TF‑IDF Params', fontsize=14)
    ax4.set_xlabel('N‑gram Range')
    ax4.set_ylabel('Max Features')

    # 5. Statistical significance heatmap (middle middle)
    ax5 = fig.add_subplot(gs[1, 1])
    sig_path = 'comprehensive_results/statistical_significance.csv'
    if os.path.exists(sig_path):
        df_sig = pd.read_csv(sig_path)
        df_sig = df_sig.rename(columns={'model_a': 'model1', 'model_b': 'model2'})
        models = sorted(set(df_sig['model1'].unique()) | set(df_sig['model2'].unique()))
        p_matrix = pd.DataFrame(index=models, columns=models, dtype=float)
        for _, row in df_sig.iterrows():
            m1, m2, p = row['model1'], row['model2'], row['p_value']
            p_matrix.loc[m1, m2] = p
            p_matrix.loc[m2, m1] = p
        np.fill_diagonal(p_matrix.values, 1.0)
        sns.heatmap(p_matrix, annot=True, fmt='.3f', cmap='RdBu_r', center=0.05,
                    cbar_kws={'label': 'p-value'}, ax=ax5)
        ax5.set_title('Statistical Significance (p‑values)', fontsize=14)
    else:
        ax5.text(0.5, 0.5, 'Statistical significance data not available',
                ha='center', va='center', fontsize=12, transform=ax5.transAxes)
        ax5.set_title('Statistical Significance (Missing)', fontsize=14)

    # 6. Misclassification matrix (middle right)
    ax6 = fig.add_subplot(gs[1, 2])
    mis_path = 'comprehensive_results/misclassified_samples.csv'
    if os.path.exists(mis_path):
        df_mis = pd.read_csv(mis_path)
        confusion_counts = df_mis.groupby(['true_class', 'predicted_class']).size().unstack(fill_value=0)
        sns.heatmap(confusion_counts, annot=True, fmt='d', cmap='Blues', ax=ax6,
                    cbar_kws={'label': 'Count'})
        ax6.set_title('Misclassification Matrix', fontsize=14)
        ax6.set_xlabel('Predicted')
        ax6.set_ylabel('True')
    else:
        ax6.text(0.5, 0.5, 'Misclassification data not available',
                ha='center', va='center', fontsize=12, transform=ax6.transAxes)
        ax6.set_title('Misclassification Matrix (Missing)', fontsize=14)

    # 7. F1 distribution box plot (bottom left)
    ax7 = fig.add_subplot(gs[2, 0])
    data = [df[df['model'] == model]['f1_mean'].values for model in df['model'].unique()]
    labels = [model.upper() for model in df['model'].unique()]
    bp = ax7.boxplot(data, labels=labels, patch_artist=True, showmeans=True,
                     meanprops={'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black'})
    colors = [MODEL_COLORS.get(model, '#95a5a6') for model in df['model'].unique()]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax7.set_ylabel('F1 Score')
    ax7.set_title('F1 Distribution Across Configurations', fontsize=14)
    ax7.tick_params(axis='x', rotation=45)
    ax7.grid(True, axis='y', alpha=0.3)

    # 8. Hyperparameter performance for best model (bottom middle)
    ax8 = fig.add_subplot(gs[2, 1])
    param_name = best_row['param_name']
    param_label = param_name.replace('_', ' ').title()
    for max_feat in sorted(best_model_df['max_features'].unique()):
        for ngram_min in sorted(best_model_df['ngram_min'].unique()):
            ngram_max = best_model_df[best_model_df['ngram_min'] == ngram_min]['ngram_max'].iloc[0]
            subset = best_model_df[(best_model_df['max_features'] == max_feat) &
                                   (best_model_df['ngram_min'] == ngram_min) &
                                   (best_model_df['ngram_max'] == ngram_max)].sort_values('param_value')
            if subset.empty:
                continue
            ax8.plot(subset['param_value'], subset['f1_mean'],
                     marker='o', label=f'mf={max_feat}, ng=({ngram_min},{ngram_max})')
    ax8.set_xlabel(param_label)
    ax8.set_ylabel('F1 Score')
    ax8.set_title(f'{best_model.upper()} F1 vs {param_label}')
    ax8.legend(fontsize=8)
    ax8.grid(True, alpha=0.3)

    # 9. Summary text (bottom right)
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    summary_text = (
        f"Best Model: {best_config}\n"
        f"F1 Score: {best_f1:.4f}\n"
        f"Accuracy: {best_row['accuracy_mean']:.4f}\n"
        f"Precision: {best_row['precision_mean']:.4f}\n"
        f"Recall: {best_row['recall_mean']:.4f}\n"
        f"Training Time: {best_row['training_duration']:.2f}s\n"
        f"Total Experiments: {len(df)}\n"
        f"Total Misclassified: {len(pd.read_csv(mis_path)) if os.path.exists(mis_path) else 'N/A'}"
    )
    ax9.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
             verticalalignment='center', horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('IndoHoaxDetector Comprehensive Analysis Report', fontsize=20, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, 'comprehensive_report.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved comprehensive report figure")

def main():
    print("Loading comprehensive experiment results...")
    df = load_data()
    print(f"Loaded {len(df)} experiment records.")

    # Create output directory
    output_dir = 'comprehensive_results/plots'
    os.makedirs(output_dir, exist_ok=True)

    print("Generating hyperparameter performance plots...")
    plot_hyperparameter_performance(df, output_dir)

    print("Generating model comparison bar chart...")
    plot_model_comparison_bar(df, output_dir)

    print("Generating TF‑IDF heatmaps...")
    plot_heatmap_tfidf(df, output_dir)

    print("Generating training time vs performance scatter...")
    plot_training_time_vs_performance(df, output_dir)

    print("Generating F1 distribution box plot...")
    plot_f1_distribution_box(df, output_dir)

    print("Generating best configurations chart...")
    plot_best_configurations(df, output_dir)

    print("Generating statistical significance visualization...")
    plot_statistical_significance(output_dir)

    print("Generating misclassification analysis plots...")
    plot_misclassification_analysis(output_dir)

    print("Generating ROC/PR curves (placeholder)...")
    plot_roc_pr_curves(output_dir)

    print("Generating comprehensive report figure...")
    plot_all_in_one_report(output_dir)

    print("\nAll plots saved to", output_dir)

if __name__ == '__main__':
    main()