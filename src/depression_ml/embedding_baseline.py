"""Optional sentence-transformer embedding + linear classifier baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from . import config
from .evaluate import best_threshold_fbeta, dump_json, evaluate_binary


def _load_encoder(model_name: str | None = None):
    name = model_name or config.EMBEDDING_MODEL_NAME
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for the embedding baseline. "
            "Install with: pip install sentence-transformers"
        ) from exc
    return SentenceTransformer(name)


def encode_texts(
    texts: pd.Series | list[str],
    *,
    model_name: str | None = None,
    batch_size: int | None = None,
) -> np.ndarray:
    encoder = _load_encoder(model_name)
    batch_size = int(batch_size if batch_size is not None else config.EMBEDDING_BATCH_SIZE)
    items = texts.tolist() if isinstance(texts, pd.Series) else list(texts)
    return encoder.encode(items, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)


def train_embedding_classifier(
    texts_train: pd.Series,
    y_train: np.ndarray,
    texts_val: pd.Series,
    y_val: np.ndarray,
    *,
    model_name: str | None = None,
) -> tuple[LogisticRegression, StandardScaler, dict[str, float], float]:
    """Fit LR on scaled MiniLM embeddings; tune threshold on validation F2."""
    X_tr = encode_texts(texts_train, model_name=model_name)
    X_va = encode_texts(texts_val, model_name=model_name)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=config.RANDOM_STATE,
    )
    clf.fit(X_tr_s, y_train)
    p_va = clf.predict_proba(X_va_s)[:, 1]
    thr = best_threshold_fbeta(y_val, p_va, beta=2.0)
    pred_va = (p_va >= thr).astype(int)
    metrics = evaluate_binary(y_val, pred_va, p_va)
    return clf, scaler, metrics, thr


def predict_embedding_proba(
    texts: pd.Series | list[str],
    clf: LogisticRegression,
    scaler: StandardScaler,
    *,
    model_name: str | None = None,
) -> np.ndarray:
    X = encode_texts(texts, model_name=model_name)
    X_s = scaler.transform(X)
    return clf.predict_proba(X_s)[:, 1]


def save_embedding_artifacts(
    clf: LogisticRegression,
    scaler: StandardScaler,
    *,
    model_name: str,
    threshold: float,
    val_metrics: dict[str, float],
    artifacts_dir: Path,
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, artifacts_dir / "embedding_classifier.pkl")
    joblib.dump(scaler, artifacts_dir / "embedding_scaler.pkl")
    (artifacts_dir / "embedding_model_name.txt").write_text(model_name, encoding="utf-8")
    dump_json(
        {
            "model_name": model_name,
            "operating_threshold": threshold,
            "val_metrics": val_metrics,
        },
        artifacts_dir / "embedding_metrics.json",
    )


def load_embedding_artifacts(artifacts_dir: Path) -> tuple[str, LogisticRegression, StandardScaler, dict[str, Any]]:
    name_path = artifacts_dir / "embedding_model_name.txt"
    if not name_path.exists():
        raise FileNotFoundError(f"Missing embedding model name: {name_path}")
    model_name = name_path.read_text(encoding="utf-8").strip()
    clf = joblib.load(artifacts_dir / "embedding_classifier.pkl")
    scaler = joblib.load(artifacts_dir / "embedding_scaler.pkl")
    meta_path = artifacts_dir / "embedding_metrics.json"
    meta = {}
    if meta_path.exists():
        import json

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return model_name, clf, scaler, meta
