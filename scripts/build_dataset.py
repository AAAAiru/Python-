#!/usr/bin/env python3
"""Build data/depression.db and train|val|test splits from one or more CSV sources."""

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
    from depression_ml.data_db import build_dataset, build_merged_dataset, dataset_stats
    from depression_ml.data_import import load_manifest, resolve_source_path

    ap = argparse.ArgumentParser(description="Build data/depression.db and train|val|test CSV splits.")
    ap.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Single source CSV (default: only with --no-all-sources)",
    )
    ap.add_argument(
        "--all-sources",
        action="store_true",
        help="Merge every available file listed in data/sources.json (recommended)",
    )
    ap.add_argument(
        "--sources",
        type=str,
        default="",
        help="Comma-separated source ids from sources.json (e.g. reddit,sentiment,urdu)",
    )
    ap.add_argument("--data-dir", type=Path, default=root / "data")
    ap.add_argument("--text-column", default="clean_text")
    ap.add_argument("--label-column", default="is_depression")
    ap.add_argument("--min-chars", type=int, default=None, help=f"Min cleaned length (default {config.MIN_DATASET_TEXT_CHARS})")
    ap.add_argument("--no-export-csv", action="store_true", help="Only write SQLite, skip train/val/test.csv")
    ap.add_argument("--include-non-english", action="store_true", help="Do not filter out non-English rows")
    ap.add_argument(
        "--split-strategy",
        choices=("source_label", "label"),
        default="source_label",
        help="Stratify by source+label when feasible, or by label only.",
    )
    ap.add_argument("--list-sources", action="store_true", help="Show which source files are present/missing")
    args = ap.parse_args()

    if args.list_sources:
        manifest = load_manifest(args.data_dir)
        for spec in manifest:
            path = resolve_source_path(args.data_dir, spec)
            status = "OK" if path else "MISSING"
            req = "required" if spec.required else "optional"
            print(f"[{status}] {spec.id} ({req}) -> {path or spec.file}")
        return

    use_merged = args.all_sources or bool(args.sources.strip())
    if use_merged:
        ids = [s.strip() for s in args.sources.split(",") if s.strip()] or None
        stats = build_merged_dataset(
            data_dir=args.data_dir,
            source_ids=ids,
            min_chars=args.min_chars,
            export_csv=not args.no_export_csv,
            english_only=not args.include_non_english,
            split_strategy=args.split_strategy,
        )
    else:
        source = args.source or (args.data_dir / "depression_dataset_reddit_cleaned.csv")
        if not source.exists():
            raise SystemExit(f"Source not found: {source}. Use --all-sources or --list-sources.")
        stats = build_dataset(
            source,
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
