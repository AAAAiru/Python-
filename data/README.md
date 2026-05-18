# Data directory

## Quick start (multi-source)

1. Place CSV files (see **`sources.json`**). At minimum you need `depression_dataset_reddit_cleaned.csv` (already in repo).
2. Optional / additional CSVs (rename after download if needed):
   - **`depression-classification-text-dataset.csv`** — already supported when present in `data/`
   - **Sentiment** → `combined_data.csv`  
     Dataset: [sentiment-analysis-for-mental-health](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health)  
     Inner file: `Combined Data.csv`
   - **Urdu** → `urdu_depression_dataset.csv`  
     Dataset: [urdu-depression-severity-dataset-2024-2025](https://www.kaggle.com/datasets/alitaqishah/urdu-depression-severity-dataset-2024-2025)
3. Build the database:

```bash
python scripts/build_dataset.py --list-sources   # see what is present
python scripts/build_dataset.py --all-sources    # merge all available → depression.db
```

## Tracked in Git (curated)

| File | Description |
|------|-------------|
| `sources.json` | Manifest of importable sources |
| `depression_dataset_reddit_cleaned.csv` | Reddit cleaned export |
| `depression.db` | SQLite (`samples` + `meta`) |
| `train.csv` / `val.csv` / `test.csv` | Stratified splits synced with DB |
| `dataset_stats.json` | Row counts, per-source breakdown |

## Label rules (binary `is_depression`)

| Source | Positive (1) | Negative (0) |
|--------|----------------|---------------|
| Reddit | `is_depression == 1` | `0` |
| Sentiment | `status == Depression` | all other statuses |
| Urdu | severity `depression_label >= 1` (mild+) | `0` = no depression |

Non-English rows are skipped by default (Urdu uses `text_english` when available). Use `--include-non-english` on build to disable.

## `data/data/` subfolder (activity logs)

The nested `data/data/control/` and `data/data/condition/` CSVs are **keyboard/mouse activity time series** (not post text). They are **not imported** into the text classifier. Use them only for separate behavioral analysis notebooks.

## Notebooks in this folder

`sentiment-analysis-for-mental-health.ipynb` and `urdu-social-media-depression-severity-detection.ipynb` are **exploration only**; training reads CSV/SQLite, not notebook outputs.

## Not in Git

- `synthetic_*.csv` — pipeline placeholders  
- Large PDFs / notebooks (unless you add them explicitly)
