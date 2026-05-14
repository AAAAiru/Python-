"""Map model scores to low / medium / high risk tiers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Tuple

import joblib
import numpy as np

from .features import vectorize_single
from .preprocess import preprocess_text_en
from .probability_calibrate import apply_platt, load_platt_optional


def load_risk_thresholds(artifacts_dir: Path) -> dict[str, float]:
    path = artifacts_dir / "risk_thresholds.json"
    if not path.exists():
        return {"low": 0.35, "high": 0.70, "operating_threshold": 0.5}
    return json.loads(path.read_text(encoding="utf-8"))


def depression_probability(
    text: str,
    model: Any,
    vectorizer: Any,
    scaler: Any,
    platt: Any | None = None,
) -> float:
    clean = preprocess_text_en(text)
    Xs = vectorize_single(vectorizer, scaler, clean)
    if hasattr(model, "predict_proba"):
        p = float(model.predict_proba(Xs)[0, 1])
    else:
        scores = model.decision_function(Xs)
        s = float(scores[0])
        p = float(1.0 / (1.0 + np.exp(-s)))
    if platt is not None:
        p = apply_platt(platt, p)
    return p


def risk_tier(prob: float, thresholds: dict[str, float]) -> str:
    low, high = float(thresholds.get("low", 0.35)), float(thresholds.get("high", 0.70))
    if prob < low:
        return "低风险"
    if prob > high:
        return "高风险"
    return "中风险"


def load_artifacts(artifacts_dir: Path):
    model = joblib.load(artifacts_dir / "depression_model.pkl")
    vectorizer = joblib.load(artifacts_dir / "tfidf.pkl")
    scaler = joblib.load(artifacts_dir / "scaler.pkl")
    thresholds = load_risk_thresholds(artifacts_dir)
    platt = load_platt_optional(artifacts_dir)
    return model, vectorizer, scaler, thresholds, platt


def assess(text: str, artifacts_dir: Path) -> Tuple[str, float, dict[str, float]]:
    model, vectorizer, scaler, thr, platt = load_artifacts(artifacts_dir)
    p = depression_probability(text, model, vectorizer, scaler, platt)
    tier = risk_tier(p, thr)
    return tier, p, thr


def assess_with_lexicon(text: str, artifacts_dir: Path) -> Tuple[str, float, dict[str, float], dict[str, float]]:
    """Same as ``assess`` plus EMNLP'17 lexicon hit counts on preprocessed text."""
    from .emnlp17_signals import extract_emnlp17_features

    model, vectorizer, scaler, thr, platt = load_artifacts(artifacts_dir)
    clean = preprocess_text_en(text)
    lex = extract_emnlp17_features(clean)
    p = depression_probability(text, model, vectorizer, scaler, platt)
    tier = risk_tier(p, thr)
    return tier, p, thr, lex
