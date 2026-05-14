"""Load Kaggle CSVs from data/ with flexible filenames and column names."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import DATA_DIR, REDDIT_BINARY_FILTER


TEXT_CANDIDATES = ("text", "statement", "content", "clean_text", "tweet", "post")
LABEL_CANDIDATES = ("status", "label", "class", "target", "mental_health", "is_depression")


@dataclass
class DatasetBundle:
    train: pd.DataFrame
    test: pd.DataFrame
    text_column: str
    label_column: str
    source_note: str


def _pick_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in df.columns:
            return name
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def _list_csv_files(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        return []
    return sorted(data_dir.glob("*.csv"))


def _read_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def _normalize_columns(df: pd.DataFrame) -> tuple[Optional[str], Optional[str]]:
    text_col = _pick_column(df, TEXT_CANDIDATES)
    label_col = _pick_column(df, LABEL_CANDIDATES)
    return text_col, label_col


def _prepare_frame(df: pd.DataFrame, text_col: str, label_col: str) -> pd.DataFrame:
    out = df[[text_col, label_col]].copy()
    out.rename(columns={text_col: "text_raw", label_col: "label_raw"}, inplace=True)
    out["text_raw"] = out["text_raw"].astype("string")
    out["label_raw"] = out["label_raw"].astype("string").str.strip()
    out = out[out["text_raw"].notna() & (out["text_raw"].str.len() > 0)]
    out = out.drop_duplicates(subset=["text_raw"])
    return out.reset_index(drop=True)


def load_from_explicit_train_test(train_path: Path, test_path: Path) -> DatasetBundle:
    tr = _read_csv(train_path)
    te = _read_csv(test_path)
    t_tr, l_tr = _normalize_columns(tr)
    t_te, l_te = _normalize_columns(te)
    if not all([t_tr, l_tr, t_te, l_te]):
        raise ValueError(f"Could not detect text/label columns in {train_path.name} / {test_path.name}")
    if {t_tr, l_tr} != {t_te, l_te}:
        # still ok if names differ but same semantic - use train's names for both
        pass
    tr_n = _prepare_frame(tr, t_tr, l_tr)
    te_n = _prepare_frame(te, t_te, l_te)
    note = f"Explicit train/test: {train_path.name}, {test_path.name}"
    return DatasetBundle(train=tr_n, test=te_n, text_column="text_raw", label_column="label_raw", source_note=note)


def load_single_file_split(path: Path, test_size: float = 0.2, random_state: int = 42) -> DatasetBundle:
    from sklearn.model_selection import train_test_split

    df = _read_csv(path)
    text_col, label_col = _normalize_columns(df)
    if not text_col or not label_col:
        raise ValueError(f"Could not detect text/label columns in {path.name}. Columns: {list(df.columns)}")
    full = _prepare_frame(df, text_col, label_col)
    tr, te = train_test_split(full, test_size=test_size, random_state=random_state, stratify=full["label_raw"])
    note = f"Single file split: {path.name}"
    return DatasetBundle(train=tr.reset_index(drop=True), test=te.reset_index(drop=True), text_column="text_raw", label_column="label_raw", source_note=note)


def auto_load(data_dir: Path | None = None) -> DatasetBundle:
    """Pick the best available layout under data/."""
    data_dir = data_dir or DATA_DIR
    files = _list_csv_files(data_dir)
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}. "
            "Download a Kaggle dataset and place CSVs here. See README.md."
        )

    names = {p.name.lower(): p for p in files}

    # Prefer cleaned Reddit depression CSV when present (over synthetic placeholders).
    for key in ("depression_dataset_reddit_cleaned.csv",):
        if key.lower() in names:
            bundle = load_single_file_split(names[key.lower()], test_size=0.2, random_state=42)
            bundle.source_note += " (depression_dataset_reddit_cleaned)"
            return bundle

    pairs = [
        ("mental_health_combined_train.csv", "mental_health_combined_test.csv"),
        ("mental_health_train.csv", "mental_health_test.csv"),
        ("synthetic_train.csv", "synthetic_test.csv"),
        ("train.csv", "test.csv"),
    ]
    for a, b in pairs:
        if a.lower() in names and b.lower() in names:
            return load_from_explicit_train_test(names[a.lower()], names[b.lower()])

    # Common typo on some mirrors of the Mental Health dataset
    typo_pairs = [
        ("mental_heath_unbanlanced.csv", "mental_health_combined_test.csv"),
        ("mental_health_unbalanced.csv", "mental_health_combined_test.csv"),
        ("mental_heath_unbanlanced.csv", "test.csv"),
    ]
    for a, b in typo_pairs:
        if a.lower() in names and b.lower() in names:
            return load_from_explicit_train_test(names[a.lower()], names[b.lower()])

    # Reddit-style single file
    for key in ("sentiment_mental_health_dataset.csv", "sentiment_mental_health.csv", "reddit_mental_health.csv"):
        if key.lower() in names:
            bundle = load_single_file_split(names[key.lower()], test_size=0.25)
            if REDDIT_BINARY_FILTER:
                mask_tr = bundle.train["label_raw"].isin(["Depression", "Normal"])
                mask_te = bundle.test["label_raw"].isin(["Depression", "Normal"])
                bundle.train = bundle.train[mask_tr].reset_index(drop=True)
                bundle.test = bundle.test[mask_te].reset_index(drop=True)
                bundle.source_note += " (filtered to Depression vs Normal)"
            return bundle

    # Fallback: one obvious large training file + small test file by row count heuristic
    csvs = [(p, len(_read_csv(p))) for p in files]
    csvs.sort(key=lambda x: x[1], reverse=True)
    if len(csvs) >= 2 and csvs[0][1] >= 10 * csvs[1][1]:
        return load_from_explicit_train_test(csvs[0][0], csvs[1][0])

    # Last resort: split the largest file
    largest = csvs[0][0]
    return load_single_file_split(largest, test_size=0.2)
