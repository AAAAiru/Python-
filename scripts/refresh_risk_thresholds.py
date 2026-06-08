"""Recompute risk_thresholds.json from validation split (no full retrain)."""

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
    import joblib
    import numpy as np

    from depression_ml import config
    from depression_ml.evaluate import derive_risk_thresholds, dump_json
    from depression_ml.features import vectorize_text_stats
    from depression_ml.io_data import auto_load
    from depression_ml.labels import binary_labels
    from depression_ml.preprocess import preprocess_text_en
    from depression_ml.probability_calibrate import apply_platt, load_platt_optional

    parser = argparse.ArgumentParser(description="Refresh GUI risk tier thresholds from validation scores.")
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--artifacts-dir", type=Path, default=root / "artifacts")
    args = parser.parse_args()

    bundle = auto_load(args.data_dir)
    if bundle.val is None or len(bundle.val) == 0:
        raise SystemExit("No validation split in dataset; rebuild DB or run training first.")

    va = bundle.val.copy()
    va["text_clean"] = va["text_raw"].map(preprocess_text_en)
    y_va = binary_labels(va, config.POSITIVE_LABELS)

    artifacts = args.artifacts_dir
    pipeline_path = artifacts / "model_pipeline.pkl"
    if pipeline_path.exists():
        model = joblib.load(pipeline_path)
        vectorizer = scaler = None
    else:
        vectorizer = joblib.load(artifacts / "tfidf.pkl")
        scaler = joblib.load(artifacts / "scaler.pkl")
        model = joblib.load(artifacts / "depression_model.pkl")
    platt = load_platt_optional(artifacts)

    if vectorizer is None:
        p_va = model.predict_proba(va["text_clean"])[:, 1]
    else:
        X_va = vectorize_text_stats(vectorizer, scaler, va["text_clean"], fit_scaler=False)
        p_va = model.predict_proba(X_va)[:, 1]
    if platt is not None:
        p_va = np.array([apply_platt(platt, float(x)) for x in p_va])

    risk = derive_risk_thresholds(
        y_va,
        p_va,
        low_default=config.RISK_LOW_DEFAULT,
        high_default=config.RISK_HIGH_DEFAULT,
    )
    risk["calibrated_on"] = "validation_split"
    risk["probability_calibration"] = "platt_validation" if platt is not None else "none"

    out = artifacts / "risk_thresholds.json"
    dump_json(risk, out)
    metrics_path = artifacts / "metrics.json"
    if metrics_path.exists():
        import json

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["risk_thresholds"] = risk
        dump_json(metrics, metrics_path)
    print(f"Wrote {out}")
    print(risk)


if __name__ == "__main__":
    main()
