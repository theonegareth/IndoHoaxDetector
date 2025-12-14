#!/usr/bin/env python3
"""
Generate a final report for the comprehensive experiments with fixed metrics.
"""

import pandas as pd
import numpy as np
from datetime import datetime

def main():
    results_dir = "comprehensive_results"
    summary_path = f"{results_dir}/comprehensive_experiment_summary_fixed.csv"
    df = pd.read_csv(summary_path)

    # Overall best
    best_overall = df.loc[df['f1_mean'].idxmax()]

    # Best per model
    models = df['model'].unique()
    best_per_model = {}
    for model in models:
        model_df = df[df['model'] == model]
        if not model_df.empty:
            best = model_df.loc[model_df['f1_mean'].idxmax()]
            best_per_model[model] = best

    # Summary statistics
    total_experiments = len(df)
    successful = df['success'].sum()
    success_rate = successful / total_experiments * 100

    # Write report
    report_path = f"{results_dir}/final_experiment_report.md"
    with open(report_path, 'w') as f:
        f.write("# Final Experiment Report (Fixed Metrics)\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Overview\n")
        f.write(f"- **Total Experiments:** {total_experiments}\n")
        f.write(f"- **Successful Experiments:** {successful}\n")
        f.write(f"- **Success Rate:** {success_rate:.1f}%\n")
        f.write(f"- **Dataset Size:** {int(df['n_samples'].iloc[0])} samples\n")
        f.write(f"- **Cross-validation Folds:** {int(df['cv_folds'].iloc[0])}\n\n")

        f.write("## Best Overall Configuration\n")
        f.write(f"- **Model:** {best_overall['model'].upper()}\n")
        f.write(f"- **Parameter:** {best_overall['param_name']} = {best_overall['param_value']}\n")
        f.write(f"- **TF-IDF:** max_features={best_overall['max_features']}, ngram_range=({best_overall['ngram_min']},{best_overall['ngram_max']})\n")
        f.write(f"- **F1 Score:** {best_overall['f1_mean']:.4f} ± {best_overall['f1_std']:.4f}\n")
        f.write(f"- **Accuracy:** {best_overall['accuracy_mean']:.4f} ± {best_overall['accuracy_std']:.4f}\n")
        f.write(f"- **Precision:** {best_overall['precision_mean']:.4f} ± {best_overall['precision_std']:.4f}\n")
        f.write(f"- **Recall:** {best_overall['recall_mean']:.4f} ± {best_overall['recall_std']:.4f}\n")
        f.write(f"- **Training Duration:** {best_overall['training_duration']:.2f} seconds\n\n")

        f.write("## Best Configuration by Model\n")
        for model, row in best_per_model.items():
            f.write(f"### {model.upper()}\n")
            f.write(f"- **Parameter:** {row['param_name']} = {row['param_value']}\n")
            f.write(f"- **TF-IDF:** max_features={row['max_features']}, ngram_range=({row['ngram_min']},{row['ngram_max']})\n")
            f.write(f"- **F1 Score:** {row['f1_mean']:.4f} ± {row['f1_std']:.4f}\n")
            f.write(f"- **Accuracy:** {row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f}\n")
            f.write(f"- **Training Duration:** {row['training_duration']:.2f} seconds\n\n")

        f.write("## Model Performance Ranking (by F1 Score)\n")
        df_sorted = df.sort_values('f1_mean', ascending=False).head(10)
        f.write("| Rank | Model | Param | Max Features | N‑gram | F1 Score | Accuracy |\n")
        f.write("|------|-------|-------|--------------|--------|----------|----------|\n")
        for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
            f.write(f"| {i} | {row['model']} | {row['param_name']}={row['param_value']} | {row['max_features']} | ({row['ngram_min']},{row['ngram_max']}) | {row['f1_mean']:.4f} ± {row['f1_std']:.4f} | {row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f} |\n")
        f.write("\n")

        f.write("## TF‑IDF Parameter Impact\n")
        tfidf_impact = df.groupby(['max_features', 'ngram_min', 'ngram_max'])['f1_mean'].agg(['mean', 'std', 'count']).reset_index()
        tfidf_impact = tfidf_impact.sort_values('mean', ascending=False)
        f.write("| Max Features | N‑gram Range | Mean F1 | Std F1 | Count |\n")
        f.write("|--------------|--------------|---------|--------|-------|\n")
        for _, row in tfidf_impact.iterrows():
            f.write(f"| {row['max_features']} | ({row['ngram_min']},{row['ngram_max']}) | {row['mean']:.4f} | {row['std']:.4f} | {int(row['count'])} |\n")
        f.write("\n")

        f.write("## Model Comparison\n")
        model_stats = df.groupby('model')['f1_mean'].agg(['mean', 'std', 'min', 'max']).reset_index()
        model_stats = model_stats.sort_values('mean', ascending=False)
        f.write("| Model | Mean F1 | Std F1 | Min F1 | Max F1 |\n")
        f.write("|-------|---------|--------|--------|--------|\n")
        for _, row in model_stats.iterrows():
            f.write(f"| {row['model'].upper()} | {row['mean']:.4f} | {row['std']:.4f} | {row['min']:.4f} | {row['max']:.4f} |\n")
        f.write("\n")

        f.write("## Files Generated\n")
        f.write("- `comprehensive_experiment_results_fixed.csv`: All experimental results with fixed metrics\n")
        f.write("- `comprehensive_experiment_summary_fixed.csv`: Successful experiments sorted by F1\n")
        f.write("- `best_configurations_fixed.csv`: Best configuration for each model\n")
        f.write("- `comprehensive_experiment_analysis.png`: Comprehensive visualizations\n")
        f.write("- `final_experiment_report.md`: This report\n\n")

        f.write("## Recommendations\n")
        f.write("1. **For production deployment**, use the best overall configuration:\n")
        f.write(f"   - Model: **{best_overall['model'].upper()}** with {best_overall['param_name']}={best_overall['param_value']}\n")
        f.write(f"   - TF‑IDF: max_features={best_overall['max_features']}, ngram_range=({best_overall['ngram_min']},{best_overall['ngram_max']})\n")
        f.write(f"   - Expected F1: **{best_overall['f1_mean']:.4f}** ± {best_overall['f1_std']:.4f}\n")
        f.write("2. **For further optimization**, consider fine‑tuning around the best parameters with a smaller grid.\n")
        f.write("3. **Consider ensemble methods** combining top‑performing models (SVM, Random Forest, Logistic Regression).\n")
        f.write("4. **Explore advanced feature engineering** (e.g., word embeddings, transformer‑based features).\n")
        f.write("5. **Collect more labeled data** to improve generalization.\n")

    print(f"Report written to {report_path}")

if __name__ == "__main__":
    main()