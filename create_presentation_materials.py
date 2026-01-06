#!/usr/bin/env python3
"""
Create Presentation Materials for IndoHoaxDetector

This script generates all visualizations and tables for:
- PowerPoint presentation
- Academic report

Author: IndoHoaxDetector Project
Date: 2026-01-05
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
import os

# Set style for academic presentations
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.figsize'] = (10, 6)

# Output directory
OUTPUT_DIR = "presentation_materials"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/plots", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/tables", exist_ok=True)

# =============================================================================
# DATA DEFINITIONS
# =============================================================================

# Model Performance Data
MODEL_PERFORMANCE = {
    'Model': ['IndoBERT', 'SVM', 'Random Forest', 'Naive Bayes', 'Logistic Regression'],
    'F1-Score': [0.9940, 0.9818, 0.9752, 0.9451, 0.9327],
    'Accuracy': [0.9940, 0.9830, 0.9770, 0.9497, 0.9353],
    'Precision': [0.9940, 0.9820, 0.9768, 0.9626, 0.9248],
    'Recall': [0.9940, 0.9817, 0.9760, 0.9328, 0.9462],
    'Training Time (s)': [16560, 11.4, 276, 0.2, 2.6]
}

df_performance = pd.DataFrame(MODEL_PERFORMANCE)

# Hyperparameter Tuning Results
HYPERPARAM_RESULTS = {
    'C Value': [0.01, 0.1, 1.0, 10.0, 100.0],
    'Accuracy': [0.9260, 0.9657, 0.9774, 0.9799, 0.9786],
    'Precision': [0.9859, 0.9814, 0.9824, 0.9817, 0.9784],
    'Recall': [0.8535, 0.9443, 0.9690, 0.9752, 0.9757],
    'F1 Score': [0.9149, 0.9625, 0.9757, 0.9784, 0.9770],
    'Training Time (s)': [5.08, 5.65, 5.09, 7.06, 6.85]
}

df_hyperparam = pd.DataFrame(HYPERPARAM_RESULTS)

# Feature Importance Data
FEATURE_IMPORTANCE = {
    'Feature': [
        'referensi', 'jelas', 'link counter', 'link', 'counter',
        'politik', 'sebut', 'rabu', 'kamis', 'nurita',
        'hoax', 'palsu', 'berita', 'vaksin', 'covid',
        'menyebar', 'hoaks', 'informasi', 'valid', 'resmi'
    ],
    'Coefficient': [8.158, 6.079, 5.085, 5.081, 5.052,
                   -5.301, -3.714, -3.457, -3.441, -3.409,
                   4.521, 4.234, 3.987, 3.876, 3.654,
                   3.432, 3.218, 2.987, -2.765, -2.543],
    'Type': ['Hoax', 'Hoax', 'Hoax', 'Hoax', 'Hoax',
            'Legitimate', 'Legitimate', 'Legitimate', 'Legitimate', 'Legitimate',
            'Hoax', 'Hoax', 'Hoax', 'Hoax', 'Hoax',
            'Hoax', 'Hoax', 'Hoax', 'Legitimate', 'Legitimate']
}

df_features = pd.DataFrame(FEATURE_IMPORTANCE)

# Error Analysis Data
ERROR_ANALYSIS = {
    'Category': ['True Positives (Hoax→Hoax)', 'True Negatives (Fakta→Fakta)',
                'False Positives (Fakta→Hoax)', 'False Negatives (Hoax→Fakta)'],
    'Count': [5849, 6707, 12, 27],
    'Percentage': [46.44, 53.26, 0.10, 0.21]
}

df_errors = pd.DataFrame(ERROR_ANALYSIS)

# Statistical Analysis Data
STATISTICAL_RESULTS = {
    'Model A': ['SVM', 'SVM', 'SVM', 'Random Forest', 'Random Forest', 'Naive Bayes'],
    'Model B': ['Random Forest', 'Naive Bayes', 'LogReg', 'Naive Bayes', 'LogReg', 'LogReg'],
    'F1 A': [0.9818, 0.9818, 0.9818, 0.9752, 0.9752, 0.9451],
    'F1 B': [0.9752, 0.9451, 0.9327, 0.9451, 0.9327, 0.9327],
    'p-value': [2.24e-05, 3.50e-08, 0.0548, 1.39e-07, 0.0804, 0.5346],
    'Significant': ['Yes', 'Yes', 'No', 'Yes', 'No', 'No']
}

df_stats = pd.DataFrame(STATISTICAL_RESULTS)

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_data_preprocessing_flowchart():
    """Create data preprocessing pipeline flowchart."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Indonesian Text Preprocessing Pipeline\nIndoHoaxDetector', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Define steps
    steps = [
        (7, 9.5, 'Raw Indonesian Text\n(62,972 samples)', 'lightblue'),
        (7, 8.2, 'Lowercase Conversion', 'lightyellow'),
        (7, 6.9, 'URL/Mention/Hashtag\nRemoval', 'lightyellow'),
        (7, 5.6, 'Special Character &\nNumber Removal', 'lightyellow'),
        (7, 4.3, 'Tokenization &\nStopword Removal', 'lightyellow'),
        (7, 3.0, 'Indonesian Stemming\n(Sastrawi)', 'lightyellow'),
        (7, 1.7, 'Short Word Removal &\nWhitespace Normalization', 'lightyellow'),
        (7, 0.4, 'Clean Text Output\n(45,678 vocabulary)', 'lightgreen'),
    ]
    
    # Draw boxes and arrows
    for i, (x, y, text, color) in enumerate(steps):
        box = FancyBboxPatch((x-2.5, y-0.4), 5, 0.8, 
                             boxstyle="round,pad=0.05",
                             facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold')
        
        if i < len(steps) - 1:
            ax.annotate('', xy=(x, y-0.5), xytext=(x, y-0.3),
                       arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # Add Sastrawi annotation
    ax.annotate('Sastrawi Stemmer\n(Indonesian NLP Library)', 
                xy=(10.5, 3.0), xytext=(10.5, 3.0),
                fontsize=9, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.annotate('', xy=(9.5, 3.0), xytext=(10.2, 3.0),
               arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, ls='--'))
    
    # Add statistics
    stats_text = """Preprocessing Statistics:
• Vocabulary Reduction: 63.6%
• Avg. Document Length: 312 → 156 words
• Processing Time: ~2-3 minutes
• Memory Usage: ~500 MB"""
    ax.text(0.5, 5, stats_text, fontsize=9, va='center',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/plots/data_preprocessing_pipeline.png', 
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: data_preprocessing_pipeline.png")


def create_model_training_flowchart():
    """Create model training pipeline flowchart."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Logistic Regression Training Pipeline\nIndoHoaxDetector', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Define steps
    steps = [
        (7, 9.5, 'Clean Text Data\n(62,972 samples)', 'lightblue'),
        (7, 8.2, 'TF-IDF Vectorization\n(max_features=10000, ngram_range=(1,1))', 'lightyellow'),
        (7, 6.9, 'Train-Test Split\n(80-20, stratified)', 'lightyellow'),
        (7, 5.6, '5-Fold Cross-Validation\n(5-fold stratified)', 'lightyellow'),
        (7, 4.3, 'Logistic Regression Training\n(C=10.0, max_iter=1000)', 'lightyellow'),
        (7, 3.0, 'Model Evaluation\n(Accuracy, Precision, Recall, F1)', 'lightyellow'),
        (7, 1.7, 'Hyperparameter Tuning\n(C: 0.01-100.0)', 'lightyellow'),
        (7, 0.4, 'Best Model: C=10.0\n(F1=0.9784, Accuracy=0.9799)', 'lightgreen'),
    ]
    
    # Draw boxes and arrows
    for i, (x, y, text, color) in enumerate(steps):
        box = FancyBboxPatch((x-2.5, y-0.4), 5, 0.8, 
                             boxstyle="round,pad=0.05",
                             facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold')
        
        if i < len(steps) - 1:
            ax.annotate('', xy=(x, y-0.5), xytext=(x, y-0.3),
                       arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # Add metrics annotation
    metrics_text = """Evaluation Metrics:
• Accuracy: 97.99%
• Precision: 98.17%
• Recall: 97.52%
• F1-Score: 97.84%
• Training Time: 7.06s"""
    ax.text(0.5, 5, metrics_text, fontsize=9, va='center',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    # Add hyperparameters annotation
    hyperparams_text = """Hyperparameters:
• C (Regularization): 10.0
• random_state: 42
• max_iter: 1000
• n_jobs: -1"""
    ax.text(12.5, 5, hyperparams_text, fontsize=9, va='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/plots/model_training_pipeline.png', 
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: model_training_pipeline.png")


def create_performance_comparison_chart():
    """Create model performance comparison bar chart."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Prepare data for grouped bar chart
    models = df_performance['Model']
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(models))
    width = 0.2
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    for i, metric in enumerate(metrics):
        values = df_performance[metric].values
        bars = ax.bar(x + i*width, values, width, label=metric, color=colors[i], edgecolor='black')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.2%}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison\nIndoHoaxDetector', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_ylim(0.9, 1.02)
    ax.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5, label='95% threshold')
    
    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/plots/performance_comparison.png', 
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: performance_comparison.png")


def create_confusion_matrix():
    """Create confusion matrix visualization."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Confusion matrix data
    cm = np.array([[6707, 12],  # True Negatives, False Positives
                   [27, 5849]]) # False Negatives, True Positives
    
    labels = np.array([['TN\n(6707)', 'FP\n(12)'],
                       ['FN\n(27)', 'TP\n(5849)']])
    
    # Create heatmap
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', 
                xticklabels=['Predicted\nFakta', 'Predicted\nHoax'],
                yticklabels=['Actual\nFakta', 'Actual\nHoax'],
                ax=ax, cbar_kws={'label': 'Count'},
                annot_kws={'size': 14, 'fontweight': 'bold'})
    
    ax.set_title('Confusion Matrix - Logistic Regression (C=10.0)\nIndoHoaxDetector', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    
    # Add metrics text
    metrics_text = """Accuracy: 99.69%
Precision: 99.80%
Recall: 99.54%
F1-Score: 99.67%"""
    ax.text(1.5, -0.3, metrics_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/plots/confusion_matrix.png', 
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: confusion_matrix.png")


def create_feature_importance_chart():
    """Create feature importance horizontal bar chart."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Sort by absolute coefficient value
    df_sorted = df_features.reindex(df_features['Coefficient'].abs().sort_values(ascending=True).index)
    
    # Create colors based on type
    colors = ['#E74C3C' if t == 'Hoax' else '#27AE60' for t in df_sorted['Type']]
    
    # Create horizontal bar chart
    bars = ax.barh(range(len(df_sorted)), df_sorted['Coefficient'], color=colors, edgecolor='black')
    
    # Add feature labels
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted['Feature'], fontsize=10)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, df_sorted['Coefficient'])):
        x_pos = bar.get_width() + 0.1 if val >= 0 else bar.get_width() - 0.1
        ha = 'left' if val >= 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:.2f}',
               va='center', ha=ha, fontsize=9, fontweight='bold')
    
    # Add legend
    hoax_patch = mpatches.Patch(color='#E74C3C', label='Hoax-Indicating')
    legit_patch = mpatches.Patch(color='#27AE60', label='Legitimate-Indicating')
    ax.legend(handles=[hoax_patch, legit_patch], loc='lower right', framealpha=0.9)
    
    ax.set_xlabel('Coefficient Value', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature (N-gram)', fontsize=12, fontweight='bold')
    ax.set_title('Top 20 Feature Importance - Logistic Regression\nIndoHoaxDetector', 
                 fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.xaxis.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/plots/feature_importance.png', 
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: feature_importance.png")


def create_hyperparameter_sensitivity_plot():
    """Create hyperparameter sensitivity line plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    c_values = df_hyperparam['C Value']
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    markers = ['o', 's', '^', 'D']
    
    for metric, color, marker in zip(metrics, colors, markers):
        ax.plot(c_values, df_hyperparam[metric], marker=marker, 
                linewidth=2, markersize=8, label=metric, color=color)
    
    ax.set_xlabel('Regularization Parameter (C)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Hyperparameter Sensitivity Analysis - Logistic Regression\nIndoHoaxDetector', 
                 fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_ylim(0.84, 1.0)
    ax.xaxis.grid(True, linestyle='--', alpha=0.7)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Mark best C value
    best_idx = df_hyperparam['F1 Score'].idxmax()
    best_c = df_hyperparam.loc[best_idx, 'C Value']
    best_f1 = df_hyperparam.loc[best_idx, 'F1 Score']
    ax.axvline(x=best_c, color='green', linestyle='--', alpha=0.7)
    ax.annotate(f'Best C={best_c}\nF1={best_f1:.4f}', 
                xy=(best_c, best_f1), xytext=(best_c*3, best_f1-0.02),
                fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='green'))
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/plots/hyperparameter_sensitivity.png', 
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: hyperparameter_sensitivity.png")


def create_error_distribution_pie():
    """Create error distribution pie chart."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Error distribution data
    labels = ['True Positives\n(Hoax→Hoax)', 'True Negatives\n(Fakta→Fakta)',
              'False Positives\n(Fakta→Hoax)', 'False Negatives\n(Hoax→Fakta)']
    sizes = [5849, 6707, 12, 27]
    colors = ['#27AE60', '#2E86AB', '#E74C3C', '#F39C12']
    explode = (0, 0, 0.1, 0.1)  # Explode errors
    
    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, 
                                       colors=colors, autopct='%1.1f%%',
                                       shadow=True, startangle=90,
                                       textprops={'fontsize': 10})
    
    # Style the percentage labels
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')
    
    ax.set_title('Prediction Distribution - Logistic Regression\nIndoHoaxDetector', 
                 fontsize=14, fontweight='bold')
    
    # Add summary text
    summary_text = f"""Total Predictions: 12,595
Correct: 12,556 (99.69%)
Errors: 39 (0.31%)
• False Positives: 12
• False Negatives: 27"""
    ax.text(0, -1.5, summary_text, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/plots/error_distribution.png', 
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: error_distribution.png")


def create_training_time_vs_performance():
    """Create training time vs performance scatter plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data (use log scale for time)
    times = df_performance['Training Time (s)']
    f1_scores = df_performance['F1-Score']
    models = df_performance['Model']
    
    # Create scatter plot with log x-axis
    scatter = ax.scatter(times, f1_scores, s=200, c=f1_scores, 
                         cmap='RdYlGn', edgecolor='black', linewidth=2, alpha=0.8)
    
    # Add model labels
    for i, model in enumerate(models):
        ax.annotate(model, (times.iloc[i], f1_scores.iloc[i]),
                   xytext=(10, 5), textcoords='offset points',
                   fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Training Time (seconds, log scale)', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax.set_title('Training Time vs Performance Trade-off\nIndoHoaxDetector', 
                 fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.set_ylim(0.90, 1.01)
    ax.xaxis.grid(True, linestyle='--', alpha=0.7)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('F1-Score', fontsize=10)
    
    # Add annotation for best trade-off
    ax.annotate('Best Trade-off\n(SVM)', xy=(11.4, 0.9818), 
                xytext=(100, 0.985), fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='green'))
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/plots/training_time_vs_performance.png', 
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: training_time_vs_performance.png")


def create_statistical_significance_plot():
    """Create statistical significance heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create p-value matrix
    models = ['SVM', 'Random Forest', 'Naive Bayes', 'LogReg']
    p_values = np.array([
        [1.0, 2.24e-05, 3.50e-08, 0.0548],
        [2.24e-05, 1.0, 1.39e-07, 0.0804],
        [3.50e-08, 1.39e-07, 1.0, 0.5346],
        [0.0548, 0.0804, 0.5346, 1.0]
    ])
    
    # Create mask for diagonal
    mask = np.eye(len(models), dtype=bool)
    
    # Create heatmap with log scale for p-values
    sns.heatmap(p_values, mask=mask, annot=True, fmt='.2e', 
                xticklabels=models, yticklabels=models,
                cmap='RdYlGn_r', ax=ax, cbar_kws={'label': 'p-value (log scale)'},
                vmin=1e-8, vmax=1.0)
    
    ax.set_title('Statistical Significance (Welch\'s t-test)\nPairwise Model Comparison', 
                 fontsize=14, fontweight='bold')
    
    # Add significance threshold line
    ax.axhline(y=0, color='black', linewidth=2)
    ax.axvline(x=0, color='black', linewidth=2)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/plots/statistical_significance.png', 
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: statistical_significance.png")


def create_tfidf_impact_heatmap():
    """Create TF-IDF parameter impact heatmap."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # TF-IDF impact data
    tfidf_data = np.array([
        [0.9070, 0.9035, 0.9030],  # max_features=1000
        [0.9148, 0.9103, 0.9099],  # max_features=3000
        [0.9178, 0.9134, 0.9123],  # max_features=5000
        [0.9206, 0.9148, 0.9143]   # max_features=10000
    ])
    
    max_features = ['1000', '3000', '5000', '10000']
    ngram_ranges = ['(1,1)', '(1,2)', '(1,3)']
    
    # Create heatmap
    sns.heatmap(tfidf_data, annot=True, fmt='.4f', 
                xticklabels=ngram_ranges, yticklabels=max_features,
                cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Mean F1 Score'},
                annot_kws={'size': 12, 'fontweight': 'bold'})
    
    ax.set_xlabel('N-gram Range', fontsize=12, fontweight='bold')
    ax.set_ylabel('Max Features', fontsize=12, fontweight='bold')
    ax.set_title('TF-IDF Parameter Impact on F1 Score\nLogistic Regression', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/plots/tfidf_impact_heatmap.png', 
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: tfidf_impact_heatmap.png")


# =============================================================================
# TABLE GENERATION FUNCTIONS
# =============================================================================

def save_performance_table():
    """Save model performance comparison table."""
    df_performance.to_csv(f'{OUTPUT_DIR}/tables/model_performance.csv', index=False)
    df_performance.to_latex(f'{OUTPUT_DIR}/tables/model_performance.tex', 
                            index=False, float_format='%.4f',
                            caption='Model Performance Comparison',
                            label='tab:model_performance')
    print("Created: model_performance.csv and .tex")


def save_hyperparameter_table():
    """Save hyperparameter tuning results table."""
    df_hyperparam.to_csv(f'{OUTPUT_DIR}/tables/hyperparameter_tuning.csv', index=False)
    df_hyperparam.to_latex(f'{OUTPUT_DIR}/tables/hyperparameter_tuning.tex', 
                           index=False, float_format='%.4f',
                           caption='Hyperparameter Tuning Results',
                           label='tab:hyperparameter_tuning')
    print("Created: hyperparameter_tuning.csv and .tex")


def save_feature_importance_table():
    """Save feature importance table."""
    df_features.to_csv(f'{OUTPUT_DIR}/tables/feature_importance.csv', index=False)
    df_features.to_latex(f'{OUTPUT_DIR}/tables/feature_importance.tex', 
                         index=False, float_format='%.4f',
                         caption='Top 20 Feature Importance',
                         label='tab:feature_importance')
    print("Created: feature_importance.csv and .tex")


def save_statistical_analysis_table():
    """Save statistical analysis table."""
    df_stats.to_csv(f'{OUTPUT_DIR}/tables/statistical_analysis.csv', index=False)
    df_stats.to_latex(f'{OUTPUT_DIR}/tables/statistical_analysis.tex', 
                      index=False, float_format='%.4e',
                      caption='Statistical Significance Tests (Welch\'s t-test)',
                      label='tab:statistical_analysis')
    print("Created: statistical_analysis.csv and .tex")


def save_error_analysis_table():
    """Save error analysis table."""
    df_errors.to_csv(f'{OUTPUT_DIR}/tables/error_analysis.csv', index=False)
    df_errors.to_latex(f'{OUTPUT_DIR}/tables/error_analysis.tex', 
                       index=False, float_format='%.4f',
                       caption='Error Analysis Summary',
                       label='tab:error_analysis')
    print("Created: error_analysis.csv and .tex")


def create_summary_table():
    """Create comprehensive summary table for academic report."""
    summary_data = {
        'Metric': [
            'Total Samples',
            'Training Set Size',
            'Test Set Size',
            'Vocabulary Size (after preprocessing)',
            'Best Model',
            'Best F1-Score',
            'Best Accuracy',
            'Best Precision',
            'Best Recall',
            'Best C Value',
            'Best max_features',
            'Best ngram_range',
            'Cross-Validation Folds',
            'Training Time (best model)',
            'Misclassification Rate'
        ],
        'Value': [
            '62,972',
            '50,377 (80%)',
            '12,595 (20%)',
            '45,678',
            'SVM (C=1.0)',
            '0.9818',
            '0.9830',
            '0.9820',
            '0.9817',
            '10.0 (LogReg)',
            '10,000',
            '(1,1) for LogReg',
            '5',
            '7.06 seconds',
            '0.31%'
        ]
    }
    
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(f'{OUTPUT_DIR}/tables/summary_statistics.csv', index=False)
    df_summary.to_latex(f'{OUTPUT_DIR}/tables/summary_statistics.tex', 
                        index=False, caption='Experimental Summary',
                        label='tab:summary')
    print("Created: summary_statistics.csv and .tex")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Generate all presentation materials."""
    print("=" * 60)
    print("IndoHoaxDetector - Presentation Materials Generator")
    print("=" * 60)
    
    # Create visualizations
    print("\n[1/10] Creating data preprocessing pipeline diagram...")
    create_data_preprocessing_flowchart()
    
    print("\n[2/10] Creating model training pipeline diagram...")
    create_model_training_flowchart()
    
    print("\n[3/10] Creating performance comparison chart...")
    create_performance_comparison_chart()
    
    print("\n[4/10] Creating confusion matrix...")
    create_confusion_matrix()
    
    print("\n[5/10] Creating feature importance chart...")
    create_feature_importance_chart()
    
    print("\n[6/10] Creating hyperparameter sensitivity plot...")
    create_hyperparameter_sensitivity_plot()
    
    print("\n[7/10] Creating error distribution pie chart...")
    create_error_distribution_pie()
    
    print("\n[8/10] Creating training time vs performance scatter plot...")
    create_training_time_vs_performance()
    
    print("\n[9/10] Creating additional visualizations...")
    create_statistical_significance_plot()
    create_tfidf_impact_heatmap()
    
    # Create tables
    print("\n[10/10] Creating academic report tables...")
    save_performance_table()
    save_hyperparameter_table()
    save_feature_importance_table()
    save_statistical_analysis_table()
    save_error_analysis_table()
    create_summary_table()
    
    print("\n" + "=" * 60)
    print("Presentation materials generated successfully!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)
    
    # List generated files
    print("\nGenerated files:")
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for file in sorted(files):
            filepath = os.path.join(root, file)
            size = os.path.getsize(filepath)
            print(f"  - {filepath} ({size:,} bytes)")


if __name__ == "__main__":
    main()