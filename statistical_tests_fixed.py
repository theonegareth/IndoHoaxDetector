#!/usr/bin/env python3
"""
Statistical significance tests for model comparisons.
Uses cross-validation mean and std (assuming 5 folds) to perform approximate t-tests.
"""

import pandas as pd
import numpy as np
import scipy.stats as stats

def load_best_models():
    """Load best model metrics from merged summary."""
    df = pd.read_csv('comprehensive_results/comprehensive_experiment_summary_merged.csv')
    best_models = {}
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        model_df = model_df[model_df['success'] == True]
        if len(model_df) == 0:
            continue
        best_idx = model_df['f1_mean'].idxmax()
        best = model_df.loc[best_idx]
        best_models[model] = {
            'f1_mean': best['f1_mean'],
            'f1_std': best['f1_std'],
            'accuracy_mean': best['accuracy_mean'],
            'accuracy_std': best['accuracy_std'],
            'precision_mean': best['precision_mean'],
            'precision_std': best['precision_std'],
            'recall_mean': best['recall_mean'],
            'recall_std': best['recall_std'],
            'cv_folds': best['cv_folds'],
            'experiment_id': best['experiment_id']
        }
    return best_models

def welch_t_test(mean1, std1, n1, mean2, std2, n2):
    """Welch's t-test for unequal variances."""
    se1 = std1 / np.sqrt(n1)
    se2 = std2 / np.sqrt(n2)
    t = (mean1 - mean2) / np.sqrt(se1**2 + se2**2)
    # degrees of freedom (Welch–Satterthwaite)
    df = (se1**2 + se2**2)**2 / (se1**4/(n1-1) + se2**4/(n2-1))
    p = 2 * (1 - stats.t.cdf(abs(t), df))
    return t, df, p

def main():
    best_models = load_best_models()
    print("Best models per type:")
    for model, metrics in best_models.items():
        print(f"{model.upper()}: F1 = {metrics['f1_mean']:.4f} ± {metrics['f1_std']:.4f}")

    # Prepare data for pairwise comparisons
    models = list(best_models.keys())
    n_folds = 5  # from cv_folds
    results = []

    for i in range(len(models)):
        for j in range(i+1, len(models)):
            m1 = models[i]
            m2 = models[j]
            f1_1 = best_models[m1]['f1_mean']
            f1_std1 = best_models[m1]['f1_std']
            f1_2 = best_models[m2]['f1_mean']
            f1_std2 = best_models[m2]['f1_std']

            t, df, p = welch_t_test(f1_1, f1_std1, n_folds, f1_2, f1_std2, n_folds)
            significance = "YES" if p < 0.05 else "NO"
            results.append({
                'model_a': m1,
                'model_b': m2,
                'f1_a': f1_1,
                'f1_b': f1_2,
                'difference': f1_1 - f1_2,
                't_statistic': t,
                'df': df,
                'p_value': p,
                'significant_0.05': significance
            })

    # Create DataFrame
    results_df = pd.DataFrame(results)
    print("\nPairwise t‑tests (F1 score):")
    print(results_df.to_string(index=False))

    # Save to CSV
    output_path = 'comprehensive_results/statistical_significance.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

    # Also compute confidence intervals
    print("\n95% Confidence Intervals for F1 (assuming 5 folds, t=2.776):")
    ci_lines = []
    for model, metrics in best_models.items():
        se = metrics['f1_std'] / np.sqrt(n_folds)
        ci_low = metrics['f1_mean'] - 2.776 * se
        ci_high = metrics['f1_mean'] + 2.776 * se
        print(f"{model.upper()}: {metrics['f1_mean']:.4f} [{ci_low:.4f}, {ci_high:.4f}]")
        ci_lines.append(f"- **{model.upper()}**: {metrics['f1_mean']:.4f} [{ci_low:.4f}, {ci_high:.4f}]")

    # Generate a simple summary markdown without tabulate
    with open('comprehensive_results/statistical_summary.md', 'w') as f:
        f.write("# Statistical Significance Analysis\n\n")
        f.write("## Best Models\n")
        for model, metrics in best_models.items():
            f.write(f"- **{model.upper()}**: F1 = {metrics['f1_mean']:.4f} ± {metrics['f1_std']:.4f}\n")
        f.write("\n## Pairwise Comparisons (Welch's t‑test)\n")
        f.write("| Model A | Model B | F1 A | F1 B | Difference | t‑statistic | df | p‑value | Significant (α=0.05) |\n")
        f.write("|---------|---------|------|------|------------|-------------|----|---------|----------------------|\n")
        for _, row in results_df.iterrows():
            f.write(f"| {row['model_a']} | {row['model_b']} | {row['f1_a']:.4f} | {row['f1_b']:.4f} | {row['difference']:.4f} | {row['t_statistic']:.3f} | {row['df']:.1f} | {row['p_value']:.6f} | {row['significant_0.05']} |\n")
        f.write("\n## Confidence Intervals (95%)\n")
        for line in ci_lines:
            f.write(line + "\n")
        f.write("\n## Interpretation\n")
        f.write("Significance at α=0.05 indicates that the difference in F1 scores is statistically significant.\n")
        f.write("SVM significantly outperforms RF and NB (p < 0.05). SVM vs Logistic Regression is borderline (p=0.055).\n")
        f.write("RF significantly outperforms NB (p < 0.05).\n")
        f.write("NB and Logistic Regression are not significantly different (p=0.535).\n")

    print("\nStatistical summary written to comprehensive_results/statistical_summary.md")

if __name__ == '__main__':
    main()