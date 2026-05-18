#!/usr/bin/env python3
"""Build cleaned SQLite database and train/val/test CSV splits from a source file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _ensure_src() -> Path:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return root


def main() -> None:
    root = _ensure_src()
    from depression_ml import config
    from depression_ml.data_db import build_dataset, dataset_stats

    ap = argparse.ArgumentParser(description="Build data/depression.db and train|val|test CSV splits.")
    ap.add_argument(
        "--source",
        type=Path,
        default=root / "data" / "depression_dataset_reddit_cleaned.csv",
        help="Source CSV with text and label columns",
    )
    ap.add_argument("--data-dir", type=Path, default=root / "data")
    ap.add_argument("--text-column", default="clean_text")
    ap.add_argument("--label-column", default="is_depression")
    ap.add_argument("--min-chars", type=int, default=None, help=f"Min cleaned length (default {config.MIN_DATASET_TEXT_CHARS})")
    ap.add_argument("--no-export-csv", action="store_true", help="Only write SQLite, skip train/val/test.csv")
    args = ap.parse_args()

    if not args.source.exists():
        raise SystemExit(f"Source not found: {args.source}")

    stats = build_dataset(
        args.source,
        data_dir=args.data_dir,
        text_col=args.text_column,
        label_col=args.label_column,
        min_chars=args.min_chars,
        train_ratio=config.DATASET_TRAIN_RATIO,
        val_ratio=config.DATASET_VAL_RATIO,
        test_ratio=config.DATASET_TEST_RATIO,
        export_csv=not args.no_export_csv,
    )
    print("Dataset build finished.")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print("\nSummary from DB:", json.dumps(dataset_stats(args.data_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
