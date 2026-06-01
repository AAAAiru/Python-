"""Task 2: in-domain TF-IDF vs embedding + OOV cross-source evaluation."""

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
    from depression_ml.model_compare import run_task2

    parser = argparse.ArgumentParser(
        description="Compare TF-IDF (full features) vs MiniLM embeddings; optional OOV holdout eval."
    )
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--artifacts-dir", type=Path, default=root / "artifacts")
    parser.add_argument("--no-oversample", action="store_true")
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Skip sentence-transformers baseline (TF-IDF + OOV TF-IDF only).",
    )
    parser.add_argument(
        "--no-oov",
        action="store_true",
        help="Skip out-of-domain evaluation (train on reddit, test other sources).",
    )
    parser.add_argument(
        "--in-domain-only",
        action="store_true",
        help="Same as --no-oov.",
    )
    args = parser.parse_args()

    run_oov = not (args.no_oov or args.in_domain_only)

    print(
        "Task 2: model comparison. First embedding run downloads MiniLM (~90MB).",
        flush=True,
    )
    import os

    if not args.no_embeddings and not os.environ.get("HF_ENDPOINT"):
        print(
            '  Tip (China): $env:HF_ENDPOINT = "https://hf-mirror.com"  or  .\\scripts\\run_task2_mirror.ps1',
            flush=True,
        )
    if run_oov:
        print(
            "  OOV trains on sources: reddit → tests: depression_text_clf, sentiment, urdu (if in DB).",
            flush=True,
        )

    summary = run_task2(
        data_dir=args.data_dir,
        artifacts_dir=args.artifacts_dir,
        oversample=not args.no_oversample,
        with_embeddings=not args.no_embeddings,
        run_oov=run_oov,
    )

    art = args.artifacts_dir
    print("\nTask 2 finished.", flush=True)
    print(f"  {art / 'model_compare_table.csv'}", flush=True)
    print(f"  {art / 'model_compare_test_f2.png'}", flush=True)
    if run_oov and summary.get("oov") and "error" not in (summary.get("oov") or {}):
        print(f"  {art / 'oov_compare_table.csv'}", flush=True)
        print(f"  {art / 'oov_metrics.json'}", flush=True)

    ind = summary["in_domain"]
    tf = ind["tfidf_full_linear_svc"]["test_metrics"]
    print(f"\nIn-domain test — tfidf_full: F2={tf['f2']:.4f}, recall={tf['recall']:.4f}")
    emb = ind.get("embedding_minilm") or {}
    if not emb.get("skipped"):
        et = emb["test_metrics"]
        print(f"In-domain test — embedding:  F2={et['f2']:.4f}, recall={et['recall']:.4f}")
        print(f"  winner (test F2): {ind.get('winner_test_f2')}")

    oov = summary.get("oov")
    if oov and isinstance(oov, dict) and "per_holdout_source" in oov:
        print("\nOOV holdout (F2):")
        for src, entry in oov["per_holdout_source"].items():
            if entry.get("status") == "skipped":
                print(f"  {src}: skipped ({entry.get('reason')})")
                continue
            tf2 = entry.get("tfidf_linear_svc", {}).get("f2")
            line = f"  {src}: tfidf F2={tf2:.4f}" if tf2 is not None else f"  {src}:"
            emb_m = entry.get("embedding_minilm", {})
            if emb_m and "f2" in emb_m:
                line += f", embedding F2={emb_m['f2']:.4f}"
            print(line)


if __name__ == "__main__":
    main()
