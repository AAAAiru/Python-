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
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--artifacts-dir", type=Path, default=root / "artifacts")
    args = parser.parse_args()

    summary = run_training(
        data_dir=args.data_dir,
        artifacts_dir=args.artifacts_dir,
        fast=not args.slow,
        oversample=not args.no_oversample,
    )
    print("Training finished.")
    print(summary)


if __name__ == "__main__":
    main()
