# Data directory

Curated files in this folder are **tracked in Git** so clones match training splits.

| File | Description |
|------|-------------|
| `depression_dataset_reddit_cleaned.csv` | Source Reddit cleaned export |
| `depression.db` | SQLite store (`samples` + `meta` tables) |
| `train.csv` / `val.csv` / `test.csv` | Stratified splits (64% / 16% / 20%), synced with the DB |
| `dataset_stats.json` | Row counts and label distribution after build |

## Rebuild locally

```bash
python scripts/build_dataset.py
```

Options: `--min-chars`, `--source`, `--data-dir`. See root `README.md`.

## Not in Git (local only)

- `synthetic_*.csv` — optional pipeline placeholders  
- `*.ipynb` — exploration notebooks  
- Large PDFs — keep local unless you explicitly add them
