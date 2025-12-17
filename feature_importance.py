#!/usr/bin/env python3
"""
Extract feature importance from the best SVM model.
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    # Paths
    model_path = 'comprehensive_results/svm_model_c1.0_mf10000_ng1-2.pkl'
    vectorizer_path = 'comprehensive_results/tfidf_vectorizer_svm_c1.0_mf10000_ng1-2.pkl'
    output_dir = 'comprehensive_results'
    os.makedirs(output_dir, exist_ok=True)

    # Load model and vectorizer
    print(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    print(f"Loading vectorizer from {vectorizer_path}")
    vectorizer = joblib.load(vectorizer_path)

    # Get coefficients and feature names
    coef = model.coef_[0]  # shape (n_features,)
    feature_names = vectorizer.get_feature_names_out()
    print(f"Number of features: {len(feature_names)}")

    # Create DataFrame
    feat_imp = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coef,
        'abs_coef': np.abs(coef)
    })
    feat_imp = feat_imp.sort_values('abs_coef', ascending=False)

    # Save top features
    top_n = 30
    top_features = feat_imp.head(top_n)
    top_features_path = os.path.join(output_dir, 'svm_top_features.csv')
    top_features.to_csv(top_features_path, index=False)
    print(f"Top {top_n} features saved to {top_features_path}")

    # Plot
    plt.figure(figsize=(12, 8))
    sns.barplot(data=top_features, x='coefficient', y='feature', palette='viridis')
    plt.title(f'Top {top_n} Feature Importance (SVM C=1.0, max_features=10000, ngram_range=(1,2))', fontsize=14)
    plt.xlabel('Coefficient (positive favors hoax, negative favors legitimate)')
    plt.ylabel('Feature (ngram)')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'svm_feature_importance.png')
    plt.savefig(plot_path, dpi=300)
    print(f"Feature importance plot saved to {plot_path}")

    # Also save all features with coefficients
    all_features_path = os.path.join(output_dir, 'svm_all_features.csv')
    feat_imp.to_csv(all_features_path, index=False)
    print(f"All features saved to {all_features_path}")

    # Print some insights
    print("\n=== Feature Importance Insights ===")
    print(f"Total features: {len(feat_imp)}")
    print(f"Most positive (hoax-indicating) features:")
    for _, row in feat_imp.sort_values('coefficient', ascending=False).head(5).iterrows():
        print(f"  {row['feature']}: {row['coefficient']:.4f}")
    print(f"Most negative (legitimate-indicating) features:")
    for _, row in feat_imp.sort_values('coefficient', ascending=True).head(5).iterrows():
        print(f"  {row['feature']}: {row['coefficient']:.4f}")

if __name__ == '__main__':
    main()