"""Post-hoc Platt scaling on validation probabilities (sklearn LogisticRegression on 1D scores)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression


def fit_platt_calibrator(p_val: np.ndarray, y_val: np.ndarray) -> LogisticRegression | None:
    """Map raw classifier scores in [0,1] to calibrated probabilities. Returns None if not fittable."""
    p_val = np.asarray(p_val, dtype=float).reshape(-1)
    y_val = np.asarray(y_val, dtype=int).reshape(-1)
    if len(p_val) < 10 or np.unique(p_val).size < 2 or np.unique(y_val).size < 2:
        return None
    lr = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
    try:
        lr.fit(p_val.reshape(-1, 1), y_val)
    except ValueError:
        return None
    return lr


def apply_platt(calibrator: Any, p_raw: float) -> float:
    return float(np.clip(calibrator.predict_proba([[float(p_raw)]])[0, 1], 1e-6, 1.0 - 1e-6))


def load_platt_optional(artifacts_dir: Path) -> LogisticRegression | None:
    path = artifacts_dir / "platt_calibrator.pkl"
    if not path.exists():
        return None
    obj = joblib.load(path)
    if isinstance(obj, LogisticRegression):
        return obj
    if isinstance(obj, dict) and obj.get("kind") == "platt":
        m = obj.get("model")
        if isinstance(m, LogisticRegression):
            return m
    return None


def save_platt(calibrator: LogisticRegression, artifacts_dir: Path) -> None:
    joblib.dump({"kind": "platt", "model": calibrator}, artifacts_dir / "platt_calibrator.pkl")
