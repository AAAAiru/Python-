"""Map model scores to low / medium / high risk tiers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np

from .features import vectorize_single
from .preprocess import preprocess_text_en


def load_risk_thresholds(artifacts_dir: Path) -> dict[str, float]:
    path = artifacts_dir / "risk_thresholds.json"
    if not path.exists():
        return {"low": 0.35, "high": 0.70, "operating_threshold": 0.5}
    return json.loads(path.read_text(encoding="utf-8"))


def depression_probability(text: str, model, vectorizer, scaler) -> float:
    clean = preprocess_text_en(text)
    Xs = vectorize_single(vectorizer, scaler, clean)
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(Xs)[0, 1])
    scores = model.decision_function(Xs)
    s = float(scores[0])
    return float(1.0 / (1.0 + np.exp(-s)))


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
    return model, vectorizer, scaler, thresholds


def assess(text: str, artifacts_dir: Path) -> Tuple[str, float, dict[str, float]]:
    model, vectorizer, scaler, thr = load_artifacts(artifacts_dir)
    p = depression_probability(text, model, vectorizer, scaler)
    tier = risk_tier(p, thr)
    return tier, p, thr
