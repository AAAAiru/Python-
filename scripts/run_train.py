from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ensure_src_on_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return root


def main() -> None:
    root = _ensure_src_on_path()
    from depression_ml.train import run_training

    parser = argparse.ArgumentParser(description="Train depression vs non-depression models.")
    parser.add_argument("--slow", action="store_true", help="Enable GridSearchCV on some models (slower).")
    parser.add_argument("--no-oversample", action="store_true", help="Disable RandomOverSampler on training features.")
    parser.add_argument(
        "--rebuild-data",
        action="store_true",
        help="Force rebuild SQLite + train|val|test CSV from data/sources.json before training.",
    )
    parser.add_argument(
        "--no-auto-rebuild",
        action="store_true",
        help="Skip automatic rebuild when new/changed source CSVs are detected.",
    )
    parser.add_argument(
        "--with-embeddings",
        action="store_true",
        help="Train sentence-transformer (MiniLM) embedding baseline and compare metrics.",
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Skip embedding baseline even if enabled in config.",
    )
    parser.add_argument(
        "--run-oov",
        action="store_true",
        help="Run out-of-domain evaluation (train on reddit, test on other sources).",
    )
    parser.add_argument(
        "--no-oov",
        action="store_true",
        help="Skip out-of-domain evaluation.",
    )
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--artifacts-dir", type=Path, default=root / "artifacts")
    args = parser.parse_args()

    with_embeddings = None
    if args.with_embeddings:
        with_embeddings = True
    elif args.no_embeddings:
        with_embeddings = False

    run_oov = None
    if args.run_oov:
        run_oov = True
    elif args.no_oov:
        run_oov = False

    summary = run_training(
        data_dir=args.data_dir,
        artifacts_dir=args.artifacts_dir,
        fast=not args.slow,
        oversample=not args.no_oversample,
        rebuild_data=args.rebuild_data,
        auto_rebuild_data=not args.no_auto_rebuild,
        with_embeddings=with_embeddings,
        run_oov=run_oov,
    )
    print("Training finished.")
    print(summary)


if __name__ == "__main__":
    main()
