"""Task 2: in-domain TF-IDF (full features) vs MiniLM embedding baseline on the same splits."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from . import config
from .embedding_baseline import (
    predict_embedding_proba,
    save_embedding_artifacts,
    train_embedding_classifier,
)
from .evaluate import best_threshold_fbeta, dump_json, evaluate_binary, plot_model_bar
from .features import fit_tfidf, vectorize_text_stats
from .io_data import DatasetBundle, auto_load
from .labels import binary_labels
from .preprocess import preprocess_text_en


def _log(msg: str) -> None:
    print(msg, flush=True)


def _prepare_splits(bundle: DatasetBundle) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = bundle.train.copy()
    test_df = bundle.test.copy()
    train_df["text_clean"] = train_df["text_raw"].map(preprocess_text_en)
    test_df["text_clean"] = test_df["text_raw"].map(preprocess_text_en)
    if bundle.val is not None and len(bundle.val) > 0:
        val_df = bundle.val.copy()
        val_df["text_clean"] = val_df["text_raw"].map(preprocess_text_en)
    else:
        y_all = binary_labels(train_df, config.POSITIVE_LABELS)
        idx = np.arange(len(train_df))
        tr_idx, va_idx = train_test_split(
            idx,
            test_size=config.VAL_FRACTION,
            random_state=config.RANDOM_STATE,
            stratify=y_all,
        )
        val_df = train_df.iloc[va_idx].reset_index(drop=True)
        train_df = train_df.iloc[tr_idx].reset_index(drop=True)
    return train_df, val_df, test_df


def _subsample(tr: pd.DataFrame, y_tr: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    cap = int(config.MAX_TRAIN_ROWS)
    if len(tr) <= cap:
        return tr, y_tr
    sub_idx, _ = train_test_split(
        np.arange(len(tr)),
        train_size=cap,
        stratify=y_tr,
        random_state=config.RANDOM_STATE,
    )
    return tr.iloc[sub_idx].reset_index(drop=True), y_tr[sub_idx]


def train_tfidf_full_linear(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    oversample: bool = True,
) -> tuple[Any, Any, Any, float, dict[str, float]]:
    y_tr = binary_labels(train_df, config.POSITIVE_LABELS)
    y_va = binary_labels(val_df, config.POSITIVE_LABELS)
    tr, y_tr = _subsample(train_df, y_tr)

    vectorizer = fit_tfidf(tr["text_clean"])
    scaler = StandardScaler(with_mean=False)
    X_tr = vectorize_text_stats(vectorizer, scaler, tr["text_clean"], fit_scaler=True)
    X_va = vectorize_text_stats(vectorizer, scaler, val_df["text_clean"], fit_scaler=False)

    if oversample and 0.01 < y_tr.mean() < 0.99:
        ros = RandomOverSampler(random_state=config.RANDOM_STATE)
        X_tr, y_tr = ros.fit_resample(X_tr, y_tr)

    model = CalibratedClassifierCV(
        LinearSVC(
            class_weight="balanced",
            dual="auto",
            max_iter=20000,
            random_state=config.RANDOM_STATE,
        ),
        method="sigmoid",
        cv=3,
    )
    model.fit(X_tr, y_tr)
    p_va = model.predict_proba(X_va)[:, 1]
    thr = best_threshold_fbeta(y_va, p_va, beta=2.0)
    pred_va = (p_va >= thr).astype(int)
    val_metrics = evaluate_binary(y_va, pred_va, p_va)
    return vectorizer, scaler, model, thr, val_metrics


def eval_tfidf_on_test(
    test_df: pd.DataFrame,
    vectorizer: Any,
    scaler: Any,
    model: Any,
    threshold: float,
) -> dict[str, float]:
    y_te = binary_labels(test_df, config.POSITIVE_LABELS)
    X_te = vectorize_text_stats(vectorizer, scaler, test_df["text_clean"], fit_scaler=False)
    p_te = model.predict_proba(X_te)[:, 1]
    pred_te = (p_te >= threshold).astype(int)
    return evaluate_binary(y_te, pred_te, p_te)


def run_in_domain_compare(
    *,
    data_dir: Path | None = None,
    artifacts_dir: Path | None = None,
    oversample: bool = True,
    save_artifacts: bool = True,
) -> dict[str, Any]:
    """Train/eval TF-IDF full + embedding on the same train|val|test as the main pipeline."""
    data_dir = data_dir or config.DATA_DIR
    artifacts_dir = artifacts_dir or config.ARTIFACTS_DIR
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    _log("[task2] loading dataset...")
    bundle = auto_load(data_dir)
    _log(f"[task2] {bundle.source_note}")

    train_df, val_df, test_df = _prepare_splits(bundle)
    y_va = binary_labels(val_df, config.POSITIVE_LABELS)
    y_te = binary_labels(test_df, config.POSITIVE_LABELS)
    _log(f"[task2] splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)} (test positives={int(y_te.sum())})")

    _log("[task2] training TF-IDF + full stats + LinearSVC (calibrated)...")
    vec, scaler, tfidf_model, tfidf_thr, tfidf_val = train_tfidf_full_linear(
        train_df, val_df, oversample=oversample
    )
    tfidf_test = eval_tfidf_on_test(test_df, vec, scaler, tfidf_model, tfidf_thr)
    _log(f"[task2]   tfidf_full test F2={tfidf_test['f2']:.4f}, recall={tfidf_test['recall']:.4f}")

    embedding_block: dict[str, Any] = {"skipped": True}
    try:
        _log(f"[task2] encoding with {config.EMBEDDING_MODEL_NAME} (first run downloads model)...")
        y_tr = binary_labels(train_df, config.POSITIVE_LABELS)
        tr_sub, y_tr_sub = _subsample(train_df, y_tr)
        emb_clf, emb_scaler, emb_val, emb_thr = train_embedding_classifier(
            tr_sub["text_clean"],
            y_tr_sub,
            val_df["text_clean"],
            y_va,
        )
        p_te_emb = predict_embedding_proba(test_df["text_clean"], emb_clf, emb_scaler)
        pred_te_emb = (p_te_emb >= emb_thr).astype(int)
        emb_test = evaluate_binary(y_te, pred_te_emb, p_te_emb)
        _log(f"[task2]   embedding test F2={emb_test['f2']:.4f}, recall={emb_test['recall']:.4f}")

        if save_artifacts:
            save_embedding_artifacts(
                emb_clf,
                emb_scaler,
                model_name=config.EMBEDDING_MODEL_NAME,
                threshold=emb_thr,
                val_metrics=emb_val,
                artifacts_dir=artifacts_dir,
            )
            joblib.dump(tfidf_model, artifacts_dir / "task2_tfidf_model.pkl")
            joblib.dump(vec, artifacts_dir / "task2_tfidf_vectorizer.pkl")
            joblib.dump(scaler, artifacts_dir / "task2_tfidf_scaler.pkl")
            dump_json(
                {"operating_threshold": tfidf_thr, "val_metrics": tfidf_val},
                artifacts_dir / "task2_tfidf_metrics.json",
            )

        embedding_block = {
            "skipped": False,
            "model_name": config.EMBEDDING_MODEL_NAME,
            "val_metrics": emb_val,
            "test_metrics": emb_test,
            "operating_threshold": emb_thr,
        }
    except (ImportError, RuntimeError, OSError) as exc:
        embedding_block = {"skipped": True, "reason": str(exc)}
        _log(f"[task2] embedding skipped: {exc}")

    report = {
        "task": "in_domain_model_compare",
        "data_note": bundle.source_note,
        "positive_labels": sorted(config.POSITIVE_LABELS),
        "tfidf_full_linear_svc": {
            "feature_set": "tfidf_full",
            "val_metrics": tfidf_val,
            "test_metrics": tfidf_test,
            "operating_threshold": tfidf_thr,
        },
        "embedding_minilm": embedding_block,
        "winner_test_f2": _pick_winner(tfidf_test, embedding_block),
    }
    return report


def _pick_winner(tfidf_test: dict[str, float], emb: dict[str, Any]) -> str:
    if emb.get("skipped"):
        return "tfidf_full_linear_svc"
    tf2 = tfidf_test["f2"]
    ef2 = emb["test_metrics"]["f2"]
    if ef2 > tf2 + 0.005:
        return "embedding_minilm"
    if tf2 > ef2 + 0.005:
        return "tfidf_full_linear_svc"
    return "tie"


def _comparison_table(report: dict[str, Any]) -> pd.DataFrame:
    rows = []
    tf = report["tfidf_full_linear_svc"]
    rows.append(
        {
            "model": "tfidf_full_linear_svc",
            "val_f2": tf["val_metrics"]["f2"],
            "test_f2": tf["test_metrics"]["f2"],
            "test_recall": tf["test_metrics"]["recall"],
            "test_f1": tf["test_metrics"]["f1"],
            "test_roc_auc": tf["test_metrics"].get("roc_auc"),
        }
    )
    emb = report.get("embedding_minilm") or {}
    if not emb.get("skipped"):
        rows.append(
            {
                "model": "embedding_minilm",
                "val_f2": emb["val_metrics"]["f2"],
                "test_f2": emb["test_metrics"]["f2"],
                "test_recall": emb["test_metrics"]["recall"],
                "test_f1": emb["test_metrics"]["f1"],
                "test_roc_auc": emb["test_metrics"].get("roc_auc"),
            }
        )
    return pd.DataFrame(rows)


def save_in_domain_report(report: dict[str, Any], artifacts_dir: Path) -> None:
    dump_json(report, artifacts_dir / "model_compare.json")
    table = _comparison_table(report)
    table.to_csv(artifacts_dir / "model_compare_table.csv", index=False, encoding="utf-8")
    bar = {}
    for _, row in table.iterrows():
        bar[str(row["model"])] = {"f2": float(row["test_f2"])}
    if bar:
        plot_model_bar(bar, artifacts_dir / "model_compare_test_f2.png", metric="f2")


def run_task2(
    *,
    data_dir: Path | None = None,
    artifacts_dir: Path | None = None,
    oversample: bool = True,
    with_embeddings: bool = True,
    run_oov: bool = True,
) -> dict[str, Any]:
    """Full task 2: in-domain compare + optional OOV holdout evaluation."""
    from .oov_eval import run_oov_eval, save_oov_report

    data_dir = data_dir or config.DATA_DIR
    artifacts_dir = artifacts_dir or config.ARTIFACTS_DIR
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    in_domain = run_in_domain_compare(
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
        oversample=oversample,
        save_artifacts=with_embeddings,
    )
    save_in_domain_report(in_domain, artifacts_dir)

    oov_report = None
    if run_oov:
        try:
            _log("[task2] OOV: train on in-domain sources, test on held-out sources...")
            oov_report = run_oov_eval(
                data_dir=data_dir,
                artifacts_dir=artifacts_dir,
                with_embeddings=with_embeddings,
            )
            save_oov_report(oov_report, artifacts_dir)
        except Exception as exc:
            oov_report = {"error": str(exc)}
            _log(f"[task2] OOV skipped/failed: {exc}")

    combined = {
        "in_domain": in_domain,
        "oov": oov_report,
    }
    dump_json(combined, artifacts_dir / "task2_summary.json")
    return combined
