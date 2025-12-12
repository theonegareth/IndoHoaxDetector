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
- `--output_dir`: Output directory (default: results/)

**Outputs:**
- `logreg_model_c{C}_mf{max_features}_ng{ngram_min}-{ngram_max}.pkl`: Trained model
- `tfidf_vectorizer_c{C}_mf{max_features}_ng{ngram_min}-{ngram_max}.pkl`: TF-IDF vectorizer
- `metrics_c{C}_mf{max_features}_ng{ngram_min}-{ngram_max}.json`: Performance metrics

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

## File Structure

```
IndoHoaxDetector/
├── train_logreg.py                    # Logistic Regression training
├── train_svm.py                       # Linear SVM training
├── train_nb.py                        # Multinomial Naive Bayes training
├── train_rf.py                        # Random Forest training
├── run_logreg_experiments.py          # Logistic Regression hyperparameter tuning
├── run_all_experiments.py             # Comprehensive multi‑model experiments
├── compare_models.py                  # Model comparison (legacy)
├── train_sklearn.py                   # Original training script (legacy)
├── train_indobert.py                  # BERT‑based training (separate)
├── preprocessed_data_FINAL_FINAL.csv  # Training data (62,972 samples)
├── results/                           # Experiment outputs
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
├── indobert_model/                    # BERT model directory
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
- `max_features`: Maximum vocabulary size (default: 5000)
- `ngram_range`: Range of n‑grams to extract (default: (1,2) → unigrams + bigrams)

## Evaluation Methodology

- **Cross‑validation**: 5‑fold stratified (preserves class distribution)
- **Metrics**: Accuracy, Precision, Recall, F1‑Score (binary, macro‑averaged)
- **Random seed**: 42 for reproducibility
- **Training/validation split**: Stratified K‑Fold

## Results Summary (Example)

### Logistic Regression (C=10.0, max_features=5000, ngram_range=(1,2))
- **Accuracy:** 0.9799 ± 0.0008
- **F1 Score:** 0.9784 ± 0.0008
- **Precision:** 0.9817 ± 0.0008
- **Recall:** 0.9752 ± 0.0019
- **Training time:** 7.06 seconds

### Best Model Selection
The `run_all_experiments.py` script automatically identifies the best‑performing model based on F1 score and provides detailed recommendations in the generated report.

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

# Load the best model (example)
model = joblib.load('results/logreg_model_c10.0_mf5000_ng1-2.pkl')
vectorizer = joblib.load('results/tfidf_vectorizer_c10.0_mf5000_ng1-2.pkl')

# Prepare text
text = ["Berita tentang kebijakan pemerintah terbaru"]
X = vectorizer.transform(text)

# Predict
prediction = model.predict(X)[0]  # 0: legitimate, 1: hoax
probabilities = model.predict_proba(X)[0]

print(f"Prediction: {'Hoax' if prediction == 1 else 'Legitimate'}")
print(f"Confidence: {probabilities[prediction]:.3f}")
```

## License

See LICENSE file for details.