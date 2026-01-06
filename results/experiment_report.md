# Logistic Regression Hyperparameter Tuning Report

**Generated:** 2025-12-11 16:29:48

## Experiment Summary

- **Total Experiments:** 5
- **Successful Experiments:** 5
- **Failed Experiments:** 0
- **Dataset Size:** 62972 samples
- **Cross-validation Folds:** 5

## Results Overview

### Best Performing Models

#### By F1 Score
- **C Value:** 10.0
- **F1 Score:** 0.9784 ± 0.0008
- **Accuracy:** 0.9799 ± 0.0008
- **Precision:** 0.9817 ± 0.0008
- **Recall:** 0.9752 ± 0.0019
- **Training Time:** 7.06 seconds

#### By Accuracy
- **C Value:** 10.0
- **Accuracy:** 0.9799 ± 0.0008
- **F1 Score:** 0.9784 ± 0.0008

### Performance Comparison

| C Value | Accuracy | Precision | Recall | F1 Score | Training Time |
|--------|----------|-----------|--------|----------|---------------|
| 0.01 | 0.9260±0.0028 | 0.9859±0.0009 | 0.8535±0.0059 | 0.9149±0.0035 | 5.08s |
| 0.1 | 0.9657±0.0017 | 0.9814±0.0017 | 0.9443±0.0027 | 0.9625±0.0019 | 5.65s |
| 1.0 | 0.9774±0.0014 | 0.9824±0.0023 | 0.9690±0.0012 | 0.9757±0.0015 | 5.09s |
| 10.0 | 0.9799±0.0008 | 0.9817±0.0008 | 0.9752±0.0019 | 0.9784±0.0008 | 7.06s |
| 100.0 | 0.9786±0.0012 | 0.9784±0.0014 | 0.9757±0.0017 | 0.9770±0.0013 | 6.85s |

## Recommendations

