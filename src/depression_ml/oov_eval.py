"""Out-of-domain evaluation: train on in-domain sources, test on held-out sources."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from . import config
from .data_db import load_all_samples
from .embedding_baseline import predict_embedding_proba, train_embedding_classifier
from .evaluate import best_threshold_fbeta, dump_json, evaluate_binary
from .features import fit_tfidf, vectorize_text_stats
from .labels import binary_labels
from .preprocess import preprocess_text_en


def _prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["text_clean"] = out["text_raw"].map(preprocess_text_en)
    return out


def _fit_tfidf_linear_svc(
    tr: pd.DataFrame,
    va: pd.DataFrame,
    y_tr: np.ndarray,
    y_va: np.ndarray,
) -> tuple[Any, Any, Any, float, dict[str, float]]:
    vectorizer = fit_tfidf(tr["text_clean"])
    scaler = StandardScaler(with_mean=False)
    X_tr = vectorize_text_stats(vectorizer, scaler, tr["text_clean"], fit_scaler=True)
    X_va = vectorize_text_stats(vectorizer, scaler, va["text_clean"], fit_scaler=False)

    model = CalibratedClassifierCV(
        LinearSVC(class_weight="balanced", dual="auto", max_iter=20000, random_state=config.RANDOM_STATE),
        method="sigmoid",
        cv=3,
    )
    model.fit(X_tr, y_tr)
    p_va = model.predict_proba(X_va)[:, 1]
    thr = best_threshold_fbeta(y_va, p_va, beta=2.0)
    pred_va = (p_va >= thr).astype(int)
    val_metrics = evaluate_binary(y_va, pred_va, p_va)
    return vectorizer, scaler, model, thr, val_metrics


def _score_tfidf(
    model: Any,
    vectorizer: Any,
    scaler: Any,
    texts: pd.Series,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    X = vectorize_text_stats(vectorizer, scaler, texts, fit_scaler=False)
    p = model.predict_proba(X)[:, 1]
    return p, (p >= threshold).astype(int)


def run_oov_eval(
    *,
    data_dir: Path | None = None,
    artifacts_dir: Path | None = None,
    train_sources: tuple[str, ...] | None = None,
    test_sources: tuple[str, ...] | None = None,
    with_embeddings: bool = True,
    val_fraction: float = 0.15,
) -> dict[str, Any]:
    """Train on OOV_TRAIN_SOURCES, evaluate per held-out source in OOV_TEST_SOURCES."""
    data_dir = data_dir or config.DATA_DIR
    artifacts_dir = artifacts_dir or config.ARTIFACTS_DIR
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_sources = train_sources or config.OOV_TRAIN_SOURCES
    test_sources = test_sources or config.OOV_TEST_SOURCES

    all_df = load_all_samples(data_dir)
    if "source_id" not in all_df.columns:
        raise ValueError("Database samples missing source_id column; rebuild with build_merged_dataset().")

    train_pool = all_df[all_df["source_id"].isin(train_sources)].reset_index(drop=True)
    if len(train_pool) < 40:
        raise ValueError(f"Only {len(train_pool)} in-domain rows for OOV train sources {train_sources}")

    train_pool = _prepare_frame(train_pool)
    y_pool = binary_labels(train_pool, config.POSITIVE_LABELS)
    idx = np.arange(len(train_pool))
    tr_idx, va_idx = train_test_split(
        idx,
        test_size=val_fraction,
        random_state=config.RANDOM_STATE,
        stratify=y_pool,
    )
    tr = train_pool.iloc[tr_idx].reset_index(drop=True)
    va = train_pool.iloc[va_idx].reset_index(drop=True)
    y_tr, y_va = y_pool[tr_idx], y_pool[va_idx]

    vec, scaler, tfidf_model, tfidf_thr, tfidf_val = _fit_tfidf_linear_svc(tr, va, y_tr, y_va)

    emb_clf = emb_scaler = None
    emb_thr = 0.5
    emb_val: dict[str, float] = {}
    if with_embeddings:
        try:
            emb_clf, emb_scaler, emb_val, emb_thr = train_embedding_classifier(
                tr["text_clean"], y_tr, va["text_clean"], y_va
            )
        except ImportError as exc:
            with_embeddings = False
            emb_val = {"error": str(exc)}

    per_source: dict[str, Any] = {}
    for src in test_sources:
        holdout = all_df[all_df["source_id"] == src]
        if len(holdout) < 5:
            per_source[src] = {"status": "skipped", "reason": "not_in_db_or_too_few_rows", "rows": len(holdout)}
            continue
        holdout = _prepare_frame(holdout.reset_index(drop=True))
        y_te = binary_labels(holdout, config.POSITIVE_LABELS)
        entry: dict[str, Any] = {"rows": len(holdout), "positives": int(y_te.sum())}

        p_tf, pred_tf = _score_tfidf(tfidf_model, vec, scaler, holdout["text_clean"], tfidf_thr)
        entry["tfidf_linear_svc"] = evaluate_binary(y_te, pred_tf, p_tf)

        if with_embeddings and emb_clf is not None and emb_scaler is not None:
            p_emb = predict_embedding_proba(holdout["text_clean"], emb_clf, emb_scaler)
            pred_emb = (p_emb >= emb_thr).astype(int)
            entry["embedding_minilm"] = evaluate_binary(y_te, pred_emb, p_emb)

        per_source[src] = entry

    in_domain_test = all_df[all_df["source_id"].isin(train_sources)]
    in_domain_test = _prepare_frame(in_domain_test.reset_index(drop=True))
    y_in = binary_labels(in_domain_test, config.POSITIVE_LABELS)
    p_in, pred_in = _score_tfidf(tfidf_model, vec, scaler, in_domain_test["text_clean"], tfidf_thr)
    in_domain_metrics = evaluate_binary(y_in, pred_in, p_in)

    report = {
        "protocol": "train_on_sources_test_on_others",
        "train_sources": list(train_sources),
        "test_sources": list(test_sources),
        "train_pool_rows": len(train_pool),
        "val_fraction": val_fraction,
        "in_domain_refit_val": tfidf_val,
        "in_domain_full_pool_test_tfidf": in_domain_metrics,
        "embedding_val": emb_val if with_embeddings else {"skipped": True},
        "per_holdout_source": per_source,
    }
    dump_json(report, artifacts_dir / "oov_metrics.json")
    return report
