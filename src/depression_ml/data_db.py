"""SQLite dataset store + build pipeline for cleaned mental-health text samples."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config
from .io_data import DatasetBundle, _prepare_frame, _read_csv
from .preprocess import preprocess_text_en


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text_raw TEXT NOT NULL,
    label_raw TEXT NOT NULL,
    text_clean TEXT NOT NULL UNIQUE,
    char_len INTEGER NOT NULL,
    split_name TEXT NOT NULL CHECK (split_name IN ('train', 'val', 'test')),
    source_id TEXT NOT NULL DEFAULT 'unknown',
    source_file TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_samples_split ON samples(split_name);
CREATE INDEX IF NOT EXISTS idx_samples_label ON samples(label_raw);
CREATE INDEX IF NOT EXISTS idx_samples_source ON samples(source_id);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def db_path(data_dir: Path | None = None) -> Path:
    return (data_dir or config.DATA_DIR) / "depression.db"


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(path: Path) -> None:
    with _connect(path) as conn:
        conn.executescript(SCHEMA_SQL)
        conn.commit()


def _set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )


def _clean_source_frame(df: pd.DataFrame, *, text_col: str, label_col: str, min_chars: int) -> pd.DataFrame:
    out = _prepare_frame(df, text_col, label_col)
    out["text_clean"] = out["text_raw"].map(preprocess_text_en)
    out["char_len"] = out["text_clean"].str.len()
    out = out[out["char_len"] >= int(min_chars)]
    out = out.drop_duplicates(subset=["text_clean"], keep="first")
    # Normalize numeric 0/1 labels to strings for storage consistency
    num = pd.to_numeric(out["label_raw"], errors="coerce")
    if num.notna().all() and set(num.unique()).issubset({0.0, 1.0}):
        out["label_raw"] = num.astype(int).astype(str)
    return out.reset_index(drop=True)


def _write_db_and_exports(
    combined: pd.DataFrame,
    *,
    data_dir: Path,
    stats_base: dict[str, Any],
    export_csv: bool,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    path = db_path(data_dir)
    with _connect(path) as conn:
        conn.execute("DROP TABLE IF EXISTS samples")
        conn.commit()
    init_db(path)

    with _connect(path) as conn:
        conn.execute("DELETE FROM meta")
        rows = [
            (
                str(r.text_raw),
                str(r.label_raw),
                str(r.text_clean),
                int(r.char_len),
                str(r.split_name),
                str(getattr(r, "source_id", "unknown")),
                str(getattr(r, "source_file", "")),
                now,
            )
            for r in combined.itertuples(index=False)
        ]
        conn.executemany(
            """
            INSERT INTO samples(
                text_raw, label_raw, text_clean, char_len, split_name, source_id, source_file, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        stats = {**stats_base, "built_at": now, "label_disclaimer": config.LABEL_DISCLAIMER}
        _set_meta(conn, "dataset_stats", json.dumps(stats, ensure_ascii=False))
        _set_meta(conn, "label_disclaimer", config.LABEL_DISCLAIMER)
        conn.commit()

    if export_csv:
        data_dir.mkdir(parents=True, exist_ok=True)
        for name in ("train", "val", "test"):
            part = combined[combined["split_name"] == name]
            export = part[["text_raw", "label_raw"]].rename(
                columns={"text_raw": "clean_text", "label_raw": "is_depression"}
            )
            export.to_csv(data_dir / f"{name}.csv", index=False, encoding="utf-8")
        (data_dir / "dataset_stats.json").write_text(
            json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    stats["db_path"] = str(path)
    return stats


def _stratified_splits(
    clean: pd.DataFrame,
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_state: int,
) -> pd.DataFrame:
    y = clean["label_raw"].astype(str)
    tr_val, te = train_test_split(
        clean,
        test_size=test_ratio,
        random_state=random_state,
        stratify=y,
    )
    val_rel = val_ratio / (train_ratio + val_ratio)
    tr, va = train_test_split(
        tr_val,
        test_size=val_rel,
        random_state=random_state,
        stratify=tr_val["label_raw"].astype(str),
    )
    for name, part in (("train", tr), ("val", va), ("test", te)):
        part["split_name"] = name
    return pd.concat([tr, va, te], ignore_index=True)


def build_merged_dataset(
    *,
    data_dir: Path | None = None,
    source_ids: list[str] | None = None,
    min_chars: int | None = None,
    train_ratio: float | None = None,
    val_ratio: float | None = None,
    test_ratio: float | None = None,
    random_state: int | None = None,
    export_csv: bool = True,
    english_only: bool = True,
) -> dict[str, Any]:
    """Merge CSVs listed in data/sources.json, split, write SQLite + train|val|test CSV."""
    from .data_import import discover_and_load

    data_dir = data_dir or config.DATA_DIR
    min_chars = int(min_chars if min_chars is not None else config.MIN_DATASET_TEXT_CHARS)
    tr = float(train_ratio if train_ratio is not None else config.DATASET_TRAIN_RATIO)
    vr = float(val_ratio if val_ratio is not None else config.DATASET_VAL_RATIO)
    te = float(test_ratio if test_ratio is not None else config.DATASET_TEST_RATIO)
    random_state = int(random_state if random_state is not None else config.RANDOM_STATE)
    if abs(tr + vr + te - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    merged, import_report = discover_and_load(
        data_dir,
        source_ids=source_ids,
        english_only=english_only,
        min_chars=min_chars,
    )
    if len(merged) < 20:
        raise ValueError(f"Only {len(merged)} rows after merge; add more source CSV files.")

    combined = _stratified_splits(merged, train_ratio=tr, val_ratio=vr, test_ratio=te, random_state=random_state)
    by_source = combined.groupby("source_id").size().astype(int).to_dict() if "source_id" in combined.columns else {}

    stats_base = {
        "mode": "merged_sources",
        "import_report": import_report,
        "rows_after_merge": int(import_report.get("rows_merged", len(merged))),
        "rows_after_split_total": len(combined),
        "min_chars": min_chars,
        "splits": {
            "train": int((combined["split_name"] == "train").sum()),
            "val": int((combined["split_name"] == "val").sum()),
            "test": int((combined["split_name"] == "test").sum()),
        },
        "per_source_in_db": by_source,
        "label_counts": combined["label_raw"].value_counts().astype(int).to_dict(),
    }
    return _write_db_and_exports(combined, data_dir=data_dir, stats_base=stats_base, export_csv=export_csv)


def build_dataset(
    source_csv: Path,
    *,
    data_dir: Path | None = None,
    text_col: str = "clean_text",
    label_col: str = "is_depression",
    min_chars: int | None = None,
    train_ratio: float = 0.64,
    val_ratio: float = 0.16,
    test_ratio: float = 0.20,
    random_state: int | None = None,
    export_csv: bool = True,
) -> dict[str, Any]:
    """Clean source CSV, stratified split, write SQLite + optional train/val/test CSV exports."""
    data_dir = data_dir or config.DATA_DIR
    min_chars = int(min_chars if min_chars is not None else config.MIN_DATASET_TEXT_CHARS)
    random_state = int(random_state if random_state is not None else config.RANDOM_STATE)
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    raw = _read_csv(source_csv)
    if text_col not in raw.columns or label_col not in raw.columns:
        raise ValueError(f"Expected columns {text_col!r}, {label_col!r}; got {list(raw.columns)}")

    before_rows = len(raw)
    clean = _clean_source_frame(raw, text_col=text_col, label_col=label_col, min_chars=min_chars)
    if len(clean) < 20:
        raise ValueError(
            f"Only {len(clean)} rows after cleaning (min_chars={min_chars}). "
            "Need more source data or lower min_chars."
        )
    clean["source_id"] = "single_csv"
    clean["source_file"] = source_csv.name

    combined = _stratified_splits(
        clean,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
    )

    stats_base = {
        "mode": "single_csv",
        "source_file": source_csv.name,
        "rows_source": before_rows,
        "rows_after_clean": len(clean),
        "rows_dropped": before_rows - len(clean),
        "min_chars": min_chars,
        "splits": {
            "train": int((combined["split_name"] == "train").sum()),
            "val": int((combined["split_name"] == "val").sum()),
            "test": int((combined["split_name"] == "test").sum()),
        },
        "label_counts": clean["label_raw"].value_counts().astype(int).to_dict(),
    }
    return _write_db_and_exports(combined, data_dir=data_dir, stats_base=stats_base, export_csv=export_csv)


def load_from_db(data_dir: Path | None = None) -> DatasetBundle:
    path = db_path(data_dir)
    if not path.exists():
        raise FileNotFoundError(path)

    with _connect(path) as conn:
        def _frame(split: str) -> pd.DataFrame:
            cur = conn.execute(
                "SELECT text_raw, label_raw FROM samples WHERE split_name=? ORDER BY id",
                (split,),
            )
            return pd.DataFrame(cur.fetchall(), columns=["text_raw", "label_raw"])

        tr, va, te = _frame("train"), _frame("val"), _frame("test")

    note = f"SQLite: {path.name} (train={len(tr)}, val={len(va)}, test={len(te)})"
    return DatasetBundle(
        train=tr,
        test=te,
        text_column="text_raw",
        label_column="label_raw",
        source_note=note,
        val=va,
    )


def dataset_stats(data_dir: Path | None = None) -> dict[str, Any]:
    path = db_path(data_dir)
    if not path.exists():
        return {}
    with _connect(path) as conn:
        row = conn.execute("SELECT value FROM meta WHERE key='dataset_stats'").fetchone()
    if not row:
        return {}
    return json.loads(row["value"])
