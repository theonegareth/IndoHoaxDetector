# IndoHoaxDetector - Machine Learning Experiments

This repository contains scripts for training and evaluating multiple machine learning models for Indonesian hoax detection using TF-IDF vectorization. The framework supports Logistic Regression, Linear SVM, Random Forest, and Multinomial Naive Bayes with configurable hyperparameters and TF-IDF parameters.

## Features

- **Multiple Models**: Logistic Regression, Linear SVM, Random Forest, Multinomial Naive Bayes
- **Hyperparameter Tuning**: Systematic experimentation across regularization strengths, tree counts, and smoothing parameters
- **Configurable TF-IDF**: Adjustable `max_features` and `ngram_range` for feature engineering
- **Comprehensive Evaluation**: 5-fold stratified cross-validation with accuracy, precision, recall, F1 score
- **Automated Reporting**: CSV results, visualizations, and detailed markdown reports
- **Model Persistence**: Save trained models and vectorizers for deployment

## Scripts

### 1. `train_logreg.py`
Train a single Logistic Regression model with specified C value and TF-IDF parameters.

```bash
python train_logreg.py --c_value 10.0 --max_features 5000 --ngram_min 1 --ngram_max 2
```

**Parameters:**
- `--c_value`: Regularization parameter C (required)
- `--max_features`: Maximum number of features for TF-IDF (default: 5000)
- `--ngram_min`: Minimum n-gram size (default: 1)
- `--ngram_max`: Maximum n-gram size (default: 2)
- `--data_path`: Path to CSV file (default: preprocessed_data_FINAL_FINAL.csv)
- `--text_col`: Text column name (default: text_clean)
- `--label_col`: Label column name (default: label_encoded)
- `--test_size`: Proportion of data for test split (default: 0.2)
- `--cv_folds`: Number of cross-validation folds (default: 5)
- `--random_state`: Random seed (default: 42)
- `--output_dir`: Output directory (default: results/train_logreg)
- `--skip_artifacts`: Skip saving model and vectorizer files

**Outputs:**
- `logreg_model_{C_sanitized}.pkl`: Trained model (C_sanitized is a filesystem-safe representation of C)
- `tfidf_vectorizer_{C_sanitized}.pkl`: TF-IDF vectorizer
- `metrics_{C_sanitized}.json`: Performance metrics (including cross-validation and test set metrics)

### 2. `train_svm.py`
Train a Linear SVM model with configurable C value.

```bash
python train_svm.py --c_value 1.0 --max_features 5000 --ngram_min 1 --ngram_max 2
```

**Parameters:** Same as `train_logreg.py` but for SVM.

### 3. `train_nb.py`
Train a Multinomial Naive Bayes model with configurable alpha (smoothing).

```bash
python train_nb.py --alpha 1.0 --max_features 5000 --ngram_min 1 --ngram_max 2
```

**Parameters:** Same as above with `--alpha` instead of `--c_value`.

### 4. `train_rf.py`
Train a Random Forest model with configurable number of trees.

```bash
python train_rf.py --n_estimators 100 --max_features 5000 --ngram_min 1 --ngram_max 2
```

**Parameters:** Same as above with `--n_estimators` for number of trees.

### 5. `run_logreg_experiments.py`
Run hyperparameter tuning experiments for Logistic Regression across multiple C values.

```bash
python run_logreg_experiments.py --c_values 0.01 0.1 1.0 10.0 100.0 --max_features 5000 --ngram_min 1 --ngram_max 2
```

**Parameters:**
- `--c_values`: List of C values to test (default: [0.01, 0.1, 1.0, 10.0, 100.0])
- `--max_features`, `--ngram_min`, `--ngram_max`: TF-IDF parameters
- `--output_dir`: Output directory (default: results/)
- `--skip_plots`: Skip generating visualization plots

**Outputs:**
- `experiment_results.csv`: Detailed results for all experiments
- `experiment_summary.csv`: Summary of successful experiments
- `hyperparameter_tuning_results.png`: Performance visualization
- `experiment_report.md`: Comprehensive experiment report
- Individual model files for each C value

### 6. `run_all_experiments.py`
Run comprehensive experiments for all models (Logistic Regression, SVM, Random Forest, Naive Bayes) with their respective hyperparameters.

```bash
python run_all_experiments.py --max_features 5000 --ngram_min 1 --ngram_max 2
```

**Parameters:**
- `--max_features`, `--ngram_min`, `--ngram_max`: TF-IDF parameters
- `--output_dir`: Output directory (default: results/)
- `--skip_plots`: Skip generating visualization plots

**Outputs:**
- `all_models_experiment_results.csv`: Detailed results for all experiments
- `all_models_experiment_summary.csv`: Summary of successful experiments
- `comprehensive_model_comparison.png`: Performance visualization across models
- `comprehensive_experiment_report.md`: Comprehensive report with analysis and recommendations
- Individual model files for each configuration

### 7. `run_comprehensive_experiments.py`
Run systematic grid search across all models and TF‑IDF parameters (max_features: 1000,3000,5000,10000; ngram_range: (1,1),(1,2),(1,3)).

```bash
python run_comprehensive_experiments.py --models logreg svm nb rf --max_parallel 2
```

**Parameters:**
- `--models`: Models to test (default: all)
- `--max_parallel`: Maximum parallel experiments (default: 2)
- `--skip_plots`: Skip generating visualization plots

**Outputs:**
- `comprehensive_experiment_results.csv`: Detailed results for all 228 experiments
- `comprehensive_experiment_summary.csv`: Summary of successful experiments
- `comprehensive_experiment_analysis.png`: Multi‑panel visualization
- `best_configurations.csv`: Best configuration for each model
- `comprehensive_experiment_report.md`: Detailed analysis and recommendations

### 8. Analysis Scripts
- `regenerate_summary.py`: Reconstruct CSV summary from JSON metric files (useful if metrics are missing).
- `analyze_experiments.py`: Generate additional visualizations and detailed analysis.
- `generate_final_report.py`: Produce a final markdown report from the comprehensive results.
- `feature_importance.py`: Extract and visualize top features from the best SVM model.
- `generate_confusion_matrices.py`: Generate confusion matrices for each model's best configuration.
- `statistical_tests_fixed.py`: Perform pairwise statistical significance tests (Welch's t‑test) on cross‑validation scores.
- `error_analysis_best_model.py`: Analyze misclassified samples from the best SVM model.

## File Structure

```
IndoHoaxDetector/
├── train_logreg.py                    # Logistic Regression training
├── train_svm.py                       # Linear SVM training
├── train_nb.py                        # Multinomial Naive Bayes training
├── train_rf.py                        # Random Forest training
├── run_logreg_experiments.py          # Logistic Regression hyperparameter tuning
├── run_all_experiments.py             # Comprehensive multi‑model experiments
├── run_comprehensive_experiments.py   # Full grid search across models & TF‑IDF
├── compare_models.py                  # Model comparison (legacy)
├── train_sklearn.py                   # Original training script (legacy)
├── train_indobert.py                  # BERT‑based training (separate)
├── feature_importance.py              # Feature importance analysis
├── generate_confusion_matrices.py     # Confusion matrix generation
├── statistical_tests_fixed.py         # Statistical significance tests
├── error_analysis_best_model.py       # Error analysis for best model
├── preprocessed_data_FINAL_FINAL.csv  # Training data (62,972 samples)
├── results/                           # Experiment outputs (ignored by git)
│   ├── experiment_results.csv
│   ├── experiment_summary.csv
│   ├── hyperparameter_tuning_results.png
│   ├── comprehensive_model_comparison.png
│   ├── experiment_report.md
│   ├── comprehensive_experiment_report.md
│   ├── logreg_model_*.pkl
│   ├── svm_model_*.pkl
│   ├── nb_model_*.pkl
│   ├── rf_model_*.pkl
│   ├── tfidf_vectorizer_*.pkl
│   └── metrics_*.json
├── comprehensive_results/             # Output of comprehensive experiments
│   ├── comprehensive_experiment_results.csv
│   ├── comprehensive_experiment_summary.csv
│   ├── comprehensive_experiment_analysis.png
│   ├── model_comparison.png
│   ├── parameter_sensitivity.png
│   ├── tfidf_impact_heatmap.png
│   ├── training_vs_performance.png
│   ├── detailed_analysis.txt
│   ├── experiment_analysis_summary.md
│   ├── final_experiment_report.md
│   ├── advanced_analysis.md           # Advanced analysis report
│   ├── confusion_matrix_best_svm.png  # Confusion matrix for best SVM
│   ├── misclassified_samples.csv      # Misclassified samples from error analysis
│   ├── svm_feature_importance.png     # Feature importance plot
│   ├── svm_top_features.csv           # Top features for SVM
│   ├── svm_all_features.csv           # All features with coefficients
│   ├── statistical_significance.csv   # Pairwise t‑test results
│   └── best_configurations.csv
├── indobert_model/                    # BERT model directory (ignored by git)
├── Huggingface_Space/                 # Deployment files for Hugging Face Space
│   ├── app.py
│   ├── requirements.txt
│   ├── README.md
│   ├── logreg_model.pkl
│   └── tfidf_vectorizer.pkl
└── README.md                          # This file
```

## Hyperparameter Configurations

| Model | Hyperparameter | Tested Values |
|-------|---------------|---------------|
| Logistic Regression | C (regularization) | [0.01, 0.1, 1.0, 10.0, 100.0] |
| Linear SVM | C (regularization) | [0.01, 0.1, 1.0, 10.0, 100.0] |
| Random Forest | n_estimators (trees) | [50, 100, 200, 500] |
| Multinomial Naive Bayes | alpha (smoothing) | [0.1, 0.5, 1.0, 2.0, 5.0] |

## TF‑IDF Vectorization

All models use the same TF‑IDF vectorization with configurable parameters:
- `max_features`: Maximum vocabulary size (tested: 1000, 3000, 5000, 10000)
- `ngram_range`: Range of n‑grams to extract (tested: (1,1), (1,2), (1,3))

## Evaluation Methodology

- **Cross‑validation**: 5‑fold stratified (preserves class distribution)
- **Metrics**: Accuracy, Precision, Recall, F1‑Score (binary, macro‑averaged)
- **Random seed**: 42 for reproducibility
- **Training/validation split**: Stratified K‑Fold

## Comprehensive Experiment Results

A systematic grid search across 4 models × 5 hyperparameter values × 4 max_features × 3 ngram_ranges = 240 configurations (228 successful) was conducted. The key findings are:

### Best Overall Configuration
- **Model**: Linear SVM
- **Hyperparameter**: C = 1.0
- **TF‑IDF**: max_features=10000, ngram_range=(1,2)
- **F1 Score**: 0.9818 ± 0.0012 (cross‑validation)
- **Test Accuracy**: 99.69%
- **Test F1**: 99.67%
- **Training time**: 11.39 seconds

### Model Performance Ranking (by average F1 Score across all configurations)
1. **SVM** – 0.9710
2. **Random Forest** – 0.9727
3. **Naive Bayes** – 0.9285
4. **Logistic Regression** – 0.9118

### TF‑IDF Impact
- **max_features**: Higher values improve performance (10000 > 5000 > 3000 > 1000)
- **ngram_range**: Bigrams (1,2) perform best overall, followed by unigrams (1,1) and trigrams (1,3).

### Training Time vs Performance
- **Fastest**: Naive Bayes (0.12–0.23 seconds) with moderate F1 (~0.91–0.94)
- **Slowest**: Random Forest (up to 323 seconds) with high F1 (~0.97)
- **Best trade‑off**: SVM (11.39 seconds) with highest F1 (0.9818)

### Advanced Analysis
The `advanced_analysis.md` report in `comprehensive_results/` provides:
- Feature importance analysis for the best SVM model
- Confusion matrices for each model's best configuration
- Statistical significance tests (Welch's t‑test) comparing models
- Error analysis of misclassified samples (only 39 out of 12,595 test samples misclassified by SVM)
- Detailed recommendations for production deployment

### Visualizations
The following plots are available in `comprehensive_results/`:
- `comprehensive_experiment_analysis.png` – Multi‑panel overview
- `model_comparison.png` – Bar chart of model performance
- `parameter_sensitivity.png` – Hyperparameter sensitivity curves
- `tfidf_impact_heatmap.png` – Heatmap of TF‑IDF parameter impact
- `training_vs_performance.png` – Training time vs F1 scatter plot
- `svm_feature_importance.png` – Top 30 features for SVM
- `confusion_matrix_best_svm.png` – Confusion matrix for best SVM

## Best Model Selection
The `run_comprehensive_experiments.py` script automatically identifies the best‑performing model based on F1 score and provides detailed recommendations in the generated report. The best model (SVM with C=1.0, max_features=10000, ngram_range=(1,2)) is recommended for production deployment.

## Dependencies

- pandas
- numpy
- scikit‑learn
- matplotlib
- seaborn
- joblib

Install with:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

## Usage Example

```python
import joblib

# Load the best model (SVM with C=1.0, max_features=10000, ngram_range=(1,2))
model = joblib.load('comprehensive_results/svm_model_c1.0_mf10000_ng1-2.pkl')
vectorizer = joblib.load('comprehensive_results/tfidf_vectorizer_svm_c1.0_mf10000_ng1-2.pkl')

# Prepare text
text = ["Berita tentang kebijakan pemerintah terbaru"]
X = vectorizer.transform(text)

# Predict
prediction = model.predict(X)[0]  # 0: legitimate, 1: hoax
probabilities = model.predict_proba(X)[0]

print(f"Prediction: {'Hoax' if prediction == 1 else 'Legitimate'}")
print(f"Confidence: {probabilities[prediction]:.3f}")
```

## Deployment

The best model can be deployed via the Hugging Face Space in `Huggingface_Space/`. Update the model and vectorizer files with the best configuration and push to the Space.

## License

See LICENSE file for details.