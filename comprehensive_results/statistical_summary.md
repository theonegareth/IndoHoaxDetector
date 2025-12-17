# Statistical Significance Analysis

## Best Models
- **SVM**: F1 = 0.9818 ± 0.0012
- **RF**: F1 = 0.9752 ± 0.0012
- **NB**: F1 = 0.9451 ± 0.0022
- **LOGREG**: F1 = 0.9327 ± 0.0408

## Pairwise Comparisons (Welch's t‑test)
| Model A | Model B | F1 A | F1 B | Difference | t‑statistic | df | p‑value | Significant (α=0.05) |
|---------|---------|------|------|------------|-------------|----|---------|----------------------|
| svm | rf | 0.9818 | 0.9752 | 0.0066 | 8.775 | 8.0 | 0.000022 | YES |
| svm | nb | 0.9818 | 0.9451 | 0.0366 | 32.770 | 6.2 | 0.000000 | YES |
| svm | logreg | 0.9818 | 0.9327 | 0.0490 | 2.687 | 4.0 | 0.054754 | NO |
| rf | nb | 0.9752 | 0.9451 | 0.0301 | 27.072 | 6.1 | 0.000000 | YES |
| rf | logreg | 0.9752 | 0.9327 | 0.0425 | 2.327 | 4.0 | 0.080427 | NO |
| nb | logreg | 0.9451 | 0.9327 | 0.0124 | 0.678 | 4.0 | 0.534574 | NO |

## Confidence Intervals (95%)
- **SVM**: 0.9818 [0.9803, 0.9833]
- **RF**: 0.9752 [0.9738, 0.9766]
- **NB**: 0.9451 [0.9424, 0.9479]
- **LOGREG**: 0.9327 [0.8821, 0.9834]

## Interpretation
Significance at α=0.05 indicates that the difference in F1 scores is statistically significant.
SVM significantly outperforms RF and NB (p < 0.05). SVM vs Logistic Regression is borderline (p=0.055).
RF significantly outperforms NB (p < 0.05).
NB and Logistic Regression are not significantly different (p=0.535).
