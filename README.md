# IndoHoaxDetector

IndoHoaxDetector is a fake news detection project focused on Indonesian-language content. It uses a traditional machine learning pipeline (TF-IDF + Logistic Regression) trained on fact-checked hoax data (e.g. TurnBackHoax) and evaluated on a preprocessed labeled dataset.

This README documents how to:
- Understand the data and model
- Run batch predictions on new data
- Evaluate the model and perform error analysis (terminal and Jupyter)
- Reuse results for an academic-style report

---

## 1. Project Structure

Key files/folders (relative to `g:/My Drive/University Files/5th Semester`):

- `Data Science/Project/IndoHoaxDetector/`
  - [`testing.ipynb`](Data Science/Project/IndoHoaxDetector/testing.ipynb:1) — batch prediction notebook for new CSVs.
  - [`evaluate_model.py`](Data Science/Project/IndoHoaxDetector/evaluate_model.py:1) — script/module to evaluate the trained model on a labeled dataset and surface error analysis examples.
  - `logreg_model.pkl` — trained Logistic Regression classifier (TF-IDF features).
  - `tfidf_vectorizer.pkl` — fitted TF-IDF vectorizer.
- `Data Science/Project/`
  - `preprocessed_data_FINAL_FINAL.csv` — main labeled dataset (preprocessed) used for evaluation.
  - Other raw/cleaned datasets and intermediate files.

Adjust paths if your layout differs.

---

## 2. Data

Main evaluation dataset (used by `evaluate_model.py`):

- File:
  - `g:/My Drive/University Files/5th Semester/Data Science/Project/preprocessed_data_FINAL_FINAL.csv`
- Important columns:
  - `text_clean`
    - Preprocessed Indonesian text:
      - lowercased
      - URLs removed
      - non-alphabetic chars removed
      - stopwords removed
      - stemmed (Sastrawi)
    - Must match the preprocessing used to train `tfidf_vectorizer.pkl`.
  - `label_encoded`
    - Ground-truth label:
      - `0` = FAKTA (real / factual)
      - `1` = HOAX (fake / misleading)

This file is treated as:
- Inputs: `text_clean`
- Targets: `label_encoded`

---

## 3. Model

Core model:

- Vectorizer:
  - TF-IDF trained on preprocessed Indonesian text.
- Classifier:
  - Logistic Regression (`logreg_model.pkl`).

The pipeline:
1. Input text (or cleaned text) is transformed with the saved TF-IDF vectorizer (`.transform`, never `.fit` or `.fit_transform` on new data).
2. Logistic Regression predicts:
   - Class: 0/1 → mapped to FAKTA/HOAX
   - Probabilities: used for confidence scores and error analysis.

---

## 4. Batch Prediction (testing.ipynb)

Use [`testing.ipynb`](Data Science/Project/IndoHoaxDetector/testing.ipynb:1) to run predictions on new, unlabeled CSV data (e.g. tweets or news):

Key configurable variables (top of notebook / script):

- `INPUT_CSV_FILE`
  - Path to your unseen CSV file.
- `TEXT_COLUMN_NAME`
  - Column with the main text (e.g. `text`).
- `TITLE_COLUMN_NAME`
  - Optional; prepended to text if present (e.g. `title`), or set to `None`.
- `OUTPUT_CSV_FILE`
  - Name for the output CSV with predictions.

Pipeline (simplified):
1. Load `logreg_model.pkl` + `tfidf_vectorizer.pkl`.
2. Load input CSV.
3. Combine title + text (if configured).
4. Apply the same cleaning function as training.
5. TF-IDF `.transform()` on cleaned text.
6. Predict:
   - `prediction` ∈ {FAKTA, HOAX}
   - `confidence_score` = max class probability.
7. Save results to `OUTPUT_CSV_FILE`.

Output columns typically include:
- original metadata (id, timestamps, etc.)
- `prediction`
- `confidence_score`

This is used for:
- Deployment-style predictions
- Qualitative analysis on real-world data.

---

## 5. Evaluation and Error Analysis (evaluate_model.py)

[`evaluate_model.py`](Data Science/Project/IndoHoaxDetector/evaluate_model.py:1) provides both:
- a CLI entrypoint for terminal usage
- a `run_evaluation()` function for Jupyter notebooks

It evaluates IndoHoaxDetector on the labeled dataset and prints:
- accuracy
- per-class precision/recall/F1
- macro/micro F1
- confusion matrix
- high-confidence false positives/false negatives (for error analysis)

### 5.1. Terminal Usage

From:

- `g:/My Drive/University Files/5th Semester/Data Science/Project/IndoHoaxDetector`

Run:

- `python evaluate_model.py`

Defaults (baked into the script):
- `--data`:
  - `g:/My Drive/University Files/5th Semester/Data Science/Project/preprocessed_data_FINAL_FINAL.csv`
- `--text-col`:
  - `text_clean`
- `--label-col`:
  - `label_encoded`
- `--model-path`:
  - `logreg_model.pkl`
- `--vectorizer-path`:
  - `tfidf_vectorizer.pkl`

You can override:

- `python evaluate_model.py --data "PATH.csv" --text-col "col" --label-col "col"`

This mode is ideal for quick, reproducible evaluation runs.

### 5.2. Jupyter Usage (Recommended for Analysis)

Do NOT use `%run evaluate_model.py` in a notebook due to ipykernel `--f` args.

Instead:

In a notebook cell inside `IndoHoaxDetector/`:

```python
from evaluate_model import run_evaluation

eval_df = run_evaluation(
    data_path="g:/My Drive/University Files/5th Semester/Data Science/Project/preprocessed_data_FINAL_FINAL.csv",
    text_col="text_clean",
    label_col="label_encoded",
    model_path="logreg_model.pkl",
    vectorizer_path="tfidf_vectorizer.pkl",
    max_show=5,
)
```

What this does:
- Loads the model and vectorizer.
- Loads the labeled dataset.
- Uses `text_clean` directly with the TF-IDF vectorizer.
- Computes and prints:
  - accuracy
  - classification report
  - confusion matrix
  - top high-confidence FP/FN examples
- Returns `eval_df` for plotting and deeper inspection.

Now you can:
- Plot confusion matrix from `eval_df`.
- Inspect FP/FN subsets:
  - `eval_df[(eval_df.true_label==0) & (eval_df.pred_label==1)]`
  - `eval_df[(eval_df.true_label==1) & (eval_df.pred_label==0)]`

---

## 6. How to Use Results in a Report

Suggested structure for your IndoHoaxDetector report:

1. Introduction
   - Motivation: fake news in Indonesia, role of automated detection.

2. Dataset
   - Describe sources and labeling.
   - Mention `preprocessed_data_FINAL_FINAL.csv` and class distribution.

3. Methodology
   - Preprocessing pipeline (as used to build `text_clean`).
   - TF-IDF feature extraction.
   - Logistic Regression classifier.

4. Experiments & Results
   - Metrics from `evaluate_model.py`:
     - accuracy
     - per-class precision/recall/F1
     - confusion matrix
   - Brief explanation:
     - how well model detects HOAX vs FAKTA.

5. Error Analysis
   - Use high-confidence FP/FN printed by `evaluate_model.py`:
     - Show representative examples (anonymized/truncated).
     - Discuss common failure patterns:
       - sensational but factual → predicted HOAX (FP)
       - subtle/neutral hoaxes → predicted FAKTA (FN)

6. Deployment Case Study
   - Use outputs from [`testing.ipynb`](Data Science/Project/IndoHoaxDetector/testing.ipynb:1) on real tweets/news:
     - HOAX rate on external data.
     - Qualitative inspection (not ground truth).

7. Ethical Considerations & Limitations
   - Misuse risks, false positives, domain shift.

8. Conclusion & Future Work
   - Summarize performance.
   - Plan improvements (e.g., IndoBERT, better calibration, more diverse data).

---

## 7. Quick Commands Summary

From `IndoHoaxDetector/`:

- Evaluate on labeled dataset (terminal):
  - `python evaluate_model.py`

- Evaluate inside Jupyter:
  - `from evaluate_model import run_evaluation`
  - `eval_df = run_evaluation()`

- Batch predict new data:
  - Open [`testing.ipynb`](Data Science/Project/IndoHoaxDetector/testing.ipynb:1)
  - Set `INPUT_CSV_FILE`, `TEXT_COLUMN_NAME`, `TITLE_COLUMN_NAME`, `OUTPUT_CSV_FILE`
  - Run all cells.

This README is tailored to your current file structure and provides everything needed to reproduce training-time preprocessing, evaluate IndoHoaxDetector, and present results rigorously.