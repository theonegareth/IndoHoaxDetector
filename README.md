# IndoHoaxDetector

Code and results for *IndoHoaxDetector: A Comparative Evaluation of Classical Machine Learning
and IndoBERT for Indonesian Mixed Corpus Hoax Detection* (ICDEES 2026).

The corpus combines four public Indonesian datasets: Deteksi Berita Hoaks Indo, Indonesia False
News Dataset 2025, Indonesian Fact and Hoax Political News, and Indonesian Hoax News Dataset.
Hoax documents in these datasets begin with fact-checker verdict tags such as `[SALAH]` and
`[HOAKS]`. A rule using only those tags reaches 99.64% weighted F1, so every result here is
computed on a sanitised corpus with the tags and fact-check apparatus removed.

## Pipeline

Run the scripts in `icdees2026/scripts/` from one folder that contains `combined_datasets_clean.csv`, which is the four
datasets concatenated with `text`, `title`, `label`, `source` and `url` columns.

| Step | Script | Output |
|---|---|---|
| Sanitise and preprocess | `deleak2.py` | `deleaked2_preproc.csv` (62,934 docs) |
| Corpus statistics | `dataset_stats.py` | `out_dataset_stats*.json` |
| Tune classical models (5-fold CV) | `classical_sanitised.py` | `out_classical_sanitised.json` |
| Metrics, CIs, efficiency | `metrics_classical.py` | `out_metrics_classical.json` |
| IndoBERT, seeds 42, 1, 2 | `train_ib_final.py SEED` | `out_ib_final_seed*.json` |
| IndoBERT, publisher-disjoint | `train_ib_final.py 42 xsource` | `out_ib_final_seed42_xsource.json` |
| Other encoders | `train_transformer.py MODEL TAG --text minimal` | `out_hf_*.json` |
| McNemar + Holm | `stats_sanitised.py` | `out_mcnemar_sanitised.csv` |
| Near-duplicates | `neardup.py` | `out_neardup.json` |
| Source-disjoint, leave-one-out | `cross_source.py` | `out_cross_source_sanitised.json` |
| Topic and temporal splits | `topic_split.py`, `temporal_split.py` | `out_topic_split.json`, `out_temporal_*.json` |
| Error analysis | `error_analysis.py` | `out_error_analysis.json` |
| Figures | `figures.py` | `fig*.png`, `fig*.pdf` |

`icdees2026/results/` holds the JSON and CSV outputs reported in the paper.

## Setup

Python 3.12, scikit-learn 1.7, PyTorch 2.6 (CUDA 12.4), Transformers 4.57.1, Sastrawi.
The reported timings come from an Intel Core i7-12700H, 31.7 GB of RAM and an NVIDIA
RTX 3050 Ti Laptop GPU (4 GB).

```
pip install -r icdees2026/requirements.txt
```

## Models

The fine-tuned IndoBERT models (`indobenchmark/indobert-base-p1`, trained on the sanitised
corpus) are at https://huggingface.co/theonegareth/IndoHoaxDetector.

## Labels and limitations

Labels come from each document's source: news portals are factual, and fact-checker or registry
documents are hoaxes. 1.18% of hoax-labelled documents carry a true, clarification or unproven
verdict. The corpus contains no social-media posts. Scores fall by 9 or more points when the test
publishers are unseen in training, so treat the results as corpus-internal.

## Repository layout

- `icdees2026/`: code and results for the paper (sanitised corpus).
- Everything else (`comprehensive_results/`, `plots/`, the notebooks and the older scripts) is the
  original coursework project. It was computed before sanitisation, and its scores are
  superseded. See [`docs/legacy_README.md`](docs/legacy_README.md).
