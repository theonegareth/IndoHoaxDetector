# Logistic Regression Regularization C Sweep Results

## Overview
This experiment evaluates the effect of the regularization parameter `C` on Logistic Regression performance for the IndoHoaxDetector classification task. The model uses TF-IDF features from the preprocessed dataset (`preprocessed_data_FINAL_FINAL.csv`). Four C values were tested: 0.01, 0.1, 1.0, and 10.0.

## Dataset
- **Total samples**: 62,972 (after cleaning)
- **Train/Test split**: 80/20 stratified split (50,377 train, 12,595 test)
- **Features**: TF-IDF with 20,000 max features (pre‑existing vectorizer)
- **Labels**: 0 = FAKTA (legitimate), 1 = HOAX (fake)

## Results

| C     | Accuracy | Precision | Recall   | F1‑Score | Training Time (s) |
|-------|----------|-----------|----------|----------|-------------------|
| 0.01  | 0.9377   | 0.9411    | 0.9377   | 0.9373   | 0.206            |
| 0.1   | 0.9657   | 0.9660    | 0.9657   | 0.9657   | 0.233            |
| 1.0   | 0.9783   | 0.9784    | 0.9783   | 0.9783   | 0.495            |
| 10.0  | **0.9813** | **0.9813** | **0.9813** | **0.9813** | 0.508            |

## Observations

1. **Accuracy improves with increasing C** (weaker regularization):
   - C = 0.01 → 93.77%
   - C = 0.1  → 96.57%
   - C = 1.0  → 97.83%
   - C = 10.0 → 98.13%

2. **Precision, recall, and F1 follow the same trend**, indicating balanced performance across both classes.

3. **Training time** increases slightly with larger C (from 0.21s to 0.51s), but remains negligible.

4. **Best performance** is achieved at **C = 10.0** with **98.13% accuracy** on the test set.

## Interpretation
- Lower C (stronger regularization) reduces overfitting but may underfit, leading to lower accuracy.
- Higher C (weaker regularization) allows the model to fit the training data more closely, improving test accuracy up to a point.
- The gains diminish beyond C = 10.0; further increases may risk overfitting but could be explored.

## Recommendations
- **Use C = 10.0** for production if computational cost is not a concern.
- For a more regularized model (e.g., when expecting noisy data), C = 1.0 still provides excellent performance (97.83%).
- Consider running a finer grid (e.g., C = [5, 10, 20, 50]) to see if accuracy plateaus or declines.

## Files Generated
- `results/logreg_c_sweep.csv` – detailed metrics per run.
- `results/logreg_c_sweep.png` – plot of accuracy, precision, recall, F1 vs C (log scale).
- `train_logreg.py` – script to train a single Logistic Regression model with a given C.
- `run_logreg_experiments.py` – script to run the sweep and generate the above outputs.

## How to Reproduce
```bash
cd IndoHoaxDetector
py run_logreg_experiments.py --c-values 0.01 0.1 1 10 --output-csv results/logreg_c_sweep.csv --plot --output-plot results/logreg_c_sweep.png
```

## Next Steps
- Compare with other models (SVM, Random Forest, Naive Bayes) using the same TF‑IDF features.
- Perform hyperparameter tuning for those models (e.g., SVM C, Random Forest max_depth).
- Conduct error analysis on misclassifications to understand model weaknesses.