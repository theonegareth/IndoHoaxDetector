# Final Experiment Report (Fixed Metrics)

**Generated:** 2025-12-14 18:42:18

## Overview
- **Total Experiments:** 60
- **Successful Experiments:** 60
- **Success Rate:** 100.0%
- **Dataset Size:** 62972 samples
- **Cross-validation Folds:** 5

## Best Overall Configuration
- **Model:** LOGREG
- **Parameter:** c_value = 10.0
- **TF-IDF:** max_features=10000, ngram_range=(1,1)
- **F1 Score:** 0.9327 ± 0.0408
- **Accuracy:** 0.9353 ± 0.0412
- **Precision:** 0.9248 ± 0.0704
- **Recall:** 0.9462 ± 0.0517
- **Training Duration:** 2.63 seconds

## Best Configuration by Model
### LOGREG
- **Parameter:** c_value = 10.0
- **TF-IDF:** max_features=10000, ngram_range=(1,1)
- **F1 Score:** 0.9327 ± 0.0408
- **Accuracy:** 0.9353 ± 0.0412
- **Training Duration:** 2.63 seconds

## Model Performance Ranking (by F1 Score)
| Rank | Model | Param | Max Features | N‑gram | F1 Score | Accuracy |
|------|-------|-------|--------------|--------|----------|----------|
| 1 | logreg | c_value=10.0 | 10000 | (1,1) | 0.9327 ± 0.0408 | 0.9353 ± 0.0412 |
| 2 | logreg | c_value=10.0 | 10000 | (1,3) | 0.9313 ± 0.0435 | 0.9341 ± 0.0437 |
| 3 | logreg | c_value=10.0 | 10000 | (1,2) | 0.9307 ± 0.0432 | 0.9332 ± 0.0439 |
| 4 | logreg | c_value=100.0 | 10000 | (1,1) | 0.9300 ± 0.0414 | 0.9322 ± 0.0428 |
| 5 | logreg | c_value=10.0 | 5000 | (1,1) | 0.9294 ± 0.0422 | 0.9324 ± 0.0423 |
| 6 | logreg | c_value=1.0 | 10000 | (1,1) | 0.9292 ± 0.0437 | 0.9331 ± 0.0416 |
| 7 | logreg | c_value=10.0 | 5000 | (1,3) | 0.9288 ± 0.0435 | 0.9315 ± 0.0445 |
| 8 | logreg | c_value=1.0 | 10000 | (1,3) | 0.9288 ± 0.0454 | 0.9331 ± 0.0428 |
| 9 | logreg | c_value=10.0 | 5000 | (1,2) | 0.9288 ± 0.0438 | 0.9313 ± 0.0451 |
| 10 | logreg | c_value=100.0 | 10000 | (1,3) | 0.9285 ± 0.0435 | 0.9311 ± 0.0440 |

## TF‑IDF Parameter Impact
| Max Features | N‑gram Range | Mean F1 | Std F1 | Count |
|--------------|--------------|---------|--------|-------|
| 10000.0 | (1.0,1.0) | 0.9206 | 0.0165 | 5 |
| 5000.0 | (1.0,1.0) | 0.9178 | 0.0172 | 5 |
| 3000.0 | (1.0,1.0) | 0.9148 | 0.0184 | 5 |
| 10000.0 | (1.0,2.0) | 0.9148 | 0.0244 | 5 |
| 10000.0 | (1.0,3.0) | 0.9143 | 0.0268 | 5 |
| 5000.0 | (1.0,2.0) | 0.9134 | 0.0250 | 5 |
| 5000.0 | (1.0,3.0) | 0.9123 | 0.0265 | 5 |
| 3000.0 | (1.0,2.0) | 0.9103 | 0.0234 | 5 |
| 3000.0 | (1.0,3.0) | 0.9099 | 0.0249 | 5 |
| 1000.0 | (1.0,1.0) | 0.9070 | 0.0190 | 5 |
| 1000.0 | (1.0,2.0) | 0.9035 | 0.0216 | 5 |
| 1000.0 | (1.0,3.0) | 0.9030 | 0.0222 | 5 |

## Model Comparison
| Model | Mean F1 | Std F1 | Min F1 | Max F1 |
|-------|---------|--------|--------|--------|
| LOGREG | 0.9118 | 0.0209 | 0.8639 | 0.9327 |

## Files Generated
- `comprehensive_experiment_results_fixed.csv`: All experimental results with fixed metrics
- `comprehensive_experiment_summary_fixed.csv`: Successful experiments sorted by F1
- `best_configurations_fixed.csv`: Best configuration for each model
- `comprehensive_experiment_analysis.png`: Comprehensive visualizations
- `final_experiment_report.md`: This report

## Recommendations
1. **For production deployment**, use the best overall configuration:
   - Model: **LOGREG** with c_value=10.0
   - TF‑IDF: max_features=10000, ngram_range=(1,1)
   - Expected F1: **0.9327** ± 0.0408
2. **For further optimization**, consider fine‑tuning around the best parameters with a smaller grid.
3. **Consider ensemble methods** combining top‑performing models (SVM, Random Forest, Logistic Regression).
4. **Explore advanced feature engineering** (e.g., word embeddings, transformer‑based features).
5. **Collect more labeled data** to improve generalization.
