#!/usr/bin/env python3
"""
Generate confusion matrices for the best configuration of each model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

def load_data():
    """Load the preprocessed dataset."""
    data_path = "preprocessed_data_FINAL_FINAL.csv"
    df = pd.read_csv(data_path)
    # Assuming columns
    text_col = "text_clean"
    label_col = "label_encoded"
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Expected columns {text_col} and {label_col} not found.")
    X = df[text_col].fillna('')
    y = df[label_col]
    return X, y

def get_best_configurations():
    """Read the summary CSV and return best config per model."""
    df = pd.read_csv('comprehensive_results/comprehensive_experiment_summary_merged.csv')
    best_configs = {}
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        # Filter successful experiments
        model_df = model_df[model_df['success'] == True]
        if len(model_df) == 0:
            continue
        best_idx = model_df['f1_mean'].idxmax()
        best = model_df.loc[best_idx]
        best_configs[model] = {
            'experiment_id': best['experiment_id'],
            'param_value': best['param_value'],
            'max_features': best['max_features'],
            'ngram_min': best['ngram_min'],
            'ngram_max': best['ngram_max'],
            'f1_mean': best['f1_mean'],
            'accuracy_mean': best['accuracy_mean'],
            'precision_mean': best['precision_mean'],
            'recall_mean': best['recall_mean']
        }
    return best_configs

def find_model_file(model, param_value, max_features, ngram_min, ngram_max):
    """Locate the saved model .pkl file."""
    base = f"comprehensive_results/"
    # Pattern: {model}_model_{param}... but naming varies.
    # Let's list files and match.
    import glob
    pattern = f"{base}/{model}_model_*mf{max_features}_ng{ngram_min}-{ngram_max}.pkl"
    files = glob.glob(pattern)
    if files:
        return files[0]
    # Try alternative pattern (some have c, a, n prefix)
    pattern2 = f"{base}/{model}_model_*{param_value}*mf{max_features}*ng{ngram_min}-{ngram_max}.pkl"
    files = glob.glob(pattern2)
    if files:
        return files[0]
    # If still not found, try generic
    pattern3 = f"{base}/{model}_model*.pkl"
    all_files = glob.glob(pattern3)
    # Filter by param value in filename
    for f in all_files:
        if f"mf{max_features}" in f and f"ng{ngram_min}-{ngram_max}" in f:
            return f
    return None

def find_vectorizer_file(model, param_value, max_features, ngram_min, ngram_max):
    """Locate the saved vectorizer .pkl file."""
    base = f"comprehensive_results/"
    import glob
    # Vectorizer naming: tfidf_vectorizer_{model}_... or tfidf_vectorizer_{param}...
    pattern = f"{base}/tfidf_vectorizer_*mf{max_features}_ng{ngram_min}-{ngram_max}.pkl"
    files = glob.glob(pattern)
    if files:
        # If multiple, pick one that matches model? Not critical.
        return files[0]
    # Try model-specific pattern
    pattern2 = f"{base}/tfidf_vectorizer_{model}_*mf{max_features}_ng{ngram_min}-{ngram_max}.pkl"
    files = glob.glob(pattern2)
    if files:
        return files[0]
    # If still not found, try generic
    pattern3 = f"{base}/tfidf_vectorizer*.pkl"
    all_files = glob.glob(pattern3)
    for f in all_files:
        if f"mf{max_features}" in f and f"ng{ngram_min}-{ngram_max}" in f:
            return f
    return None

def main():
    X, y = load_data()
    print(f"Loaded {len(X)} samples.")
    
    best_configs = get_best_configurations()
    print("Best configurations per model:")
    for model, config in best_configs.items():
        print(f"  {model}: {config}")
    
    output_dir = "comprehensive_results/confusion_matrices"
    os.makedirs(output_dir, exist_ok=True)
    
    # We'll use a 80-20 split for evaluation (same random seed for consistency)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    for model, config in best_configs.items():
        print(f"\n--- Processing {model} ---")
        model_file = find_model_file(
            model,
            config['param_value'],
            config['max_features'],
            config['ngram_min'],
            config['ngram_max']
        )
        vectorizer_file = find_vectorizer_file(
            model,
            config['param_value'],
            config['max_features'],
            config['ngram_min'],
            config['ngram_max']
        )
        if model_file is None or vectorizer_file is None:
            print(f"  Could not find model or vectorizer for {model}. Skipping.")
            continue
        
        print(f"  Model file: {model_file}")
        print(f"  Vectorizer file: {vectorizer_file}")
        
        # Load vectorizer and model
        vectorizer = joblib.load(vectorizer_file)
        clf = joblib.load(model_file)
        
        # Transform test data
        X_test_vec = vectorizer.transform(X_test)
        y_pred = clf.predict(X_test_vec)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Legitimate', 'Hoax'])
        disp.plot(cmap='Blues')
        plt.title(f'Confusion Matrix - {model.upper()}\n'
                  f'C={config["param_value"] if model=="svm" else config["param_value"]} '
                  f'max_features={config["max_features"]} ngram=({config["ngram_min"]},{config["ngram_max"]})')
        
        # Save figure
        filename = f"{output_dir}/cm_{model}_best.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"  Saved confusion matrix to {filename}")
        
        # Also compute metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='binary')
        rec = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        print(f"  Test Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
        
        # Save metrics to a CSV
        metrics_df = pd.DataFrame([{
            'model': model,
            'param_value': config['param_value'],
            'max_features': config['max_features'],
            'ngram_range': f"({config['ngram_min']},{config['ngram_max']})",
            'test_accuracy': acc,
            'test_precision': prec,
            'test_recall': rec,
            'test_f1': f1,
            'cv_f1_mean': config['f1_mean'],
            'cv_accuracy_mean': config['accuracy_mean']
        }])
        metrics_path = f"{output_dir}/metrics_{model}_best.csv"
        metrics_df.to_csv(metrics_path, index=False)
        print(f"  Metrics saved to {metrics_path}")
    
    print("\nAll confusion matrices generated.")

if __name__ == '__main__':
    main()