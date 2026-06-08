#!/usr/bin/env python3
"""Batch inference: read a CSV text column, write probabilities and risk tiers."""

from __future__ import annotations

import argparse
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
    from depression_ml.io_data import TEXT_CANDIDATES, _pick_column, _read_csv
    from depression_ml.risk import depression_probability, load_artifacts, risk_tier

    ap = argparse.ArgumentParser(description="Run depression screening on a CSV column.")
    ap.add_argument("--input", "-i", type=Path, required=True, help="Input CSV path")
    ap.add_argument("--output", "-o", type=Path, required=True, help="Output CSV path")
    ap.add_argument("--artifacts", type=Path, default=root / "artifacts", help="Artifacts directory")
    ap.add_argument("--text-column", type=str, default=None, help="Text column name (auto-detect if omitted)")
    args = ap.parse_args()

    df = _read_csv(args.input)
    text_col = args.text_column or _pick_column(df, TEXT_CANDIDATES)
    if not text_col:
        raise SystemExit(f"No text column found. Columns: {list(df.columns)}. Use --text-column.")

    model, vectorizer, scaler, thr, platt = load_artifacts(args.artifacts)
    metadata_path = args.artifacts / "model_metadata.json"
    model_version = "unknown"
    if metadata_path.exists():
        import json

        model_version = json.loads(metadata_path.read_text(encoding="utf-8")).get(
            "model_version", "unknown"
        )
    probs: list[float] = []
    tiers: list[str] = []
    for raw in df[text_col].astype("string").fillna(""):
        p = depression_probability(str(raw), model, vectorizer, scaler, platt)
        probs.append(p)
        tiers.append(risk_tier(p, thr))

    out = df.copy()
    out["model_score"] = probs
    out["risk_tier"] = tiers
    out["model_version"] = model_version
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Wrote {len(out)} rows to {args.output}")


if __name__ == "__main__":
    main()
