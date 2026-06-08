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
    from depression_ml.oov_eval import run_oov_eval

    parser = argparse.ArgumentParser(
        description="Out-of-domain eval: train on reddit (default), test on other data sources."
    )
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--artifacts-dir", type=Path, default=root / "artifacts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Skip MiniLM embedding baseline in OOV report.",
    )
    args = parser.parse_args()

    report = run_oov_eval(
        data_dir=args.data_dir,
        artifacts_dir=args.artifacts_dir,
        with_embeddings=not args.no_embeddings,
        random_state=args.seed,
    )
    print("OOV evaluation finished.")
    print(f"Wrote {args.artifacts_dir / 'oov_metrics.json'}")
    for src, metrics in report.get("per_holdout_source", {}).items():
        print(f"  {src}: {metrics}")


if __name__ == "__main__":
    main()
