#!/usr/bin/env python3
"""
Error analysis for the best SVM model.
Loads the best model and vectorizer, evaluates on test set, and extracts misclassified samples.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data():
    """Load the preprocessed dataset."""
    data_path = "preprocessed_data_FINAL_FINAL.csv"
    df = pd.read_csv(data_path)
    text_col = "text_clean"
    label_col = "label_encoded"
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Expected columns {text_col} and {label_col} not found.")
    X = df[text_col].fillna('')
    y = df[label_col]
    return X, y

def load_best_model():
    """Load the best SVM model and vectorizer."""
    model_path = "comprehensive_results/svm_model_c1.0_mf10000_ng1-2.pkl"
    vectorizer_path = "comprehensive_results/tfidf_vectorizer_svm_c1.0_mf10000_ng1-2.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def main():
    print("Loading data...")
    X, y = load_data()
    print(f"Total samples: {len(X)}")
    
    # Split into train and test (same split as used in confusion matrix generation)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Test set size: {len(X_test)}")
    
    print("Loading best SVM model and vectorizer...")
    model, vectorizer = load_best_model()
    
    print("Transforming test data...")
    X_test_vec = vectorizer.transform(X_test)
    
    print("Predicting...")
    y_pred = model.predict(X_test_vec)
    
    # Compute metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='binary')
    rec = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    
    print("\n=== Test Set Performance ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Misclassified indices
    misclassified = np.where(y_test != y_pred)[0]
    print(f"\nNumber of misclassified samples: {len(misclassified)} ({len(misclassified)/len(y_test)*100:.2f}%)")
    
    if len(misclassified) > 0:
        # Create a DataFrame for analysis
        misclassified_df = pd.DataFrame({
            'text': X_test.iloc[misclassified].values,
            'true_label': y_test.iloc[misclassified].values,
            'predicted_label': y_pred[misclassified]
        })
        # Map labels
        misclassified_df['true_class'] = misclassified_df['true_label'].map({0: 'Legitimate', 1: 'Hoax'})
        misclassified_df['predicted_class'] = misclassified_df['predicted_label'].map({0: 'Legitimate', 1: 'Hoax'})
        
        # Save to CSV
        output_path = "comprehensive_results/misclassified_samples.csv"
        misclassified_df.to_csv(output_path, index=False)
        print(f"Misclassified samples saved to {output_path}")
        
        # Analyze misclassification patterns
        print("\n=== Misclassification Breakdown ===")
        misclassification_counts = misclassified_df.groupby(['true_class', 'predicted_class']).size().unstack(fill_value=0)
        print(misclassification_counts)
        
        # Sample a few misclassified texts
        print("\n=== Sample Misclassified Texts ===")
        for i in range(min(5, len(misclassified_df))):
            row = misclassified_df.iloc[i]
            print(f"\nSample {i+1}:")
            print(f"True: {row['true_class']}, Predicted: {row['predicted_class']}")
            print(f"Text (first 200 chars): {row['text'][:200]}...")
    
    # Feature importance analysis (if model supports)
    if hasattr(model, 'coef_'):
        print("\n=== Top 10 Features for Hoax vs Legitimate ===")
        feature_names = vectorizer.get_feature_names_out()
        coefficients = model.coef_[0]
        # Sort by absolute coefficient
        top_indices = np.argsort(np.abs(coefficients))[-10:][::-1]
        for idx in top_indices:
            print(f"{feature_names[idx]}: {coefficients[idx]:.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legitimate', 'Hoax'], 
                yticklabels=['Legitimate', 'Hoax'])
    plt.title('Confusion Matrix - Best SVM Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('comprehensive_results/confusion_matrix_best_svm.png', dpi=300)
    print("\nConfusion matrix plot saved to comprehensive_results/confusion_matrix_best_svm.png")
    
    # Classification report
    print("\n=== Detailed Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Hoax']))

if __name__ == '__main__':
    main()