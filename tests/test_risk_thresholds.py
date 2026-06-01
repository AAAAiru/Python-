"""Risk tier threshold derivation."""

from __future__ import annotations

import numpy as np


def test_derive_risk_thresholds_separates_classes():
    from depression_ml.evaluate import derive_risk_thresholds

    rng = np.random.default_rng(42)
    neg = rng.uniform(0.02, 0.25, size=400)
    pos = rng.uniform(0.75, 0.98, size=400)
    y = np.array([0] * 400 + [1] * 400)
    p = np.concatenate([neg, pos])

    thr = derive_risk_thresholds(y, p, low_default=0.35, high_default=0.70)
    assert thr["low"] <= 0.35
    assert thr["high"] >= 0.70
    assert thr["low"] < thr["high"]


def test_normalize_pathological_thresholds():
    from depression_ml.risk import normalize_risk_thresholds, risk_tier

    fixed = normalize_risk_thresholds({"low": 0.044, "high": 0.973})
    assert fixed["low"] == 0.35
    assert fixed["high"] == 0.70
    assert risk_tier(0.20, {"low": 0.044, "high": 0.973}) == "低风险"
    assert risk_tier(0.80, {"low": 0.044, "high": 0.973}) == "高风险"
