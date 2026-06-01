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
    from depression_ml.ablation import run_ablation

    parser = argparse.ArgumentParser(
        description="Feature ablation + optional group-by-source split comparison."
    )
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--artifacts-dir", type=Path, default=root / "artifacts")
    parser.add_argument("--no-oversample", action="store_true", help="Disable RandomOverSampler.")
    parser.add_argument("--cv-folds", type=int, default=None, help="Stratified CV folds (default: config).")
    parser.add_argument(
        "--no-split-compare",
        action="store_true",
        help="Skip stratified_row vs group_by_source comparison (needs depression.db with 2+ sources).",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip split-protocol compare and use 2-fold CV (quicker smoke test).",
    )
    args = parser.parse_args()

    cv_folds = 2 if args.fast else args.cv_folds
    compare_splits = not args.no_split_compare and not args.fast

    print(
        "Starting ablation (no output for a while is normal — each feature set retrains TF-IDF + LR + CV).",
        flush=True,
    )
    if args.fast:
        print("  mode: --fast (2-fold CV, no split compare)", flush=True)
    else:
        print("  mode: full (5-fold CV + split compare); expect ~15–40 min on laptop", flush=True)

    report = run_ablation(
        data_dir=args.data_dir,
        artifacts_dir=args.artifacts_dir,
        oversample=not args.no_oversample,
        cv_folds=cv_folds,
        compare_splits=compare_splits,
    )
    print("Ablation finished.")
    print(f"  report: {args.artifacts_dir / 'ablation_report.json'}")
    print(f"  table:  {args.artifacts_dir / 'ablation_table.csv'}")
    for fs, res in report["feature_ablation"].items():
        print(f"  {fs}: test F2={res['test']['f2']:.4f}, test recall={res['test']['recall']:.4f}")
    if report.get("split_protocol_compare"):
        print("  split protocol (tfidf_full):")
        for proto, res in report["split_protocol_compare"].items():
            print(f"    {proto}: test F2={res['test']['f2']:.4f}")


if __name__ == "__main__":
    main()
