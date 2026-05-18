"""Train baseline + tree models, pick best on validation, calibrate risk thresholds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from . import config
from .evaluate import (
    best_threshold_fbeta,
    classification_report_dict,
    dump_json,
    evaluate_binary,
    plot_calibration_reliability,
    plot_confusion,
    plot_model_bar,
    plot_pr,
    plot_roc,
)
from .features import fit_tfidf, vectorize_text_stats
from .io_data import DatasetBundle, auto_load
from .preprocess import preprocess_text_en
from . import probability_calibrate as pcal


def _binary_labels(df: pd.DataFrame, positive_labels: set[str]) -> np.ndarray:
    col = df["label_raw"]
    num = pd.to_numeric(col, errors="coerce")
    if num.notna().all():
        u = np.unique(num.to_numpy(dtype=float))
        if u.size <= 2 and np.all(np.isin(u, (0.0, 1.0))):
            return (num.to_numpy(dtype=float) == 1.0).astype(int)
    return col.isin(positive_labels).astype(int).to_numpy()


def _maybe_subsample(tr: pd.DataFrame, y_tr: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    cap = int(config.MAX_TRAIN_ROWS)
    if len(tr) <= cap:
        return tr, y_tr
    sub_idx, _ = train_test_split(
        np.arange(len(tr)),
        train_size=cap,
        stratify=y_tr,
        random_state=config.RANDOM_STATE,
    )
    tr_sub = tr.iloc[sub_idx].reset_index(drop=True)
    y_sub = y_tr[sub_idx]
    return tr_sub, y_sub


def _build_models(fast: bool) -> dict[str, Any]:
    models: dict[str, Any] = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
            solver="saga",
        ),
        "linear_svc": CalibratedClassifierCV(
            LinearSVC(
                class_weight="balanced",
                dual="auto",
                max_iter=20000,
                random_state=config.RANDOM_STATE,
            ),
            method="sigmoid",
            cv=3,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=config.RANDOM_STATE,
            tree_method="hist",
            eval_metric="logloss",
        )
    except Exception:
        pass

    if not fast:
        models["logistic_regression"] = GridSearchCV(
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="saga",
                random_state=config.RANDOM_STATE,
            ),
            param_grid={"C": [0.25, 0.5, 1.0, 2.0]},
            scoring="f1",
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=config.RANDOM_STATE),
            n_jobs=-1,
        )
        models["random_forest"] = GridSearchCV(
            RandomForestClassifier(class_weight="balanced_subsample", random_state=config.RANDOM_STATE, n_jobs=-1),
            param_grid={"n_estimators": [200, 400], "max_depth": [None, 16, 24]},
            scoring="f1",
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=config.RANDOM_STATE),
            n_jobs=-1,
        )
    return models


def run_training(
    *,
    data_dir: Path | None = None,
    artifacts_dir: Path | None = None,
    fast: bool = True,
    oversample: bool = True,
    val_fraction: float | None = None,
) -> dict[str, Any]:
    data_dir = data_dir or config.DATA_DIR
    artifacts_dir = artifacts_dir or config.ARTIFACTS_DIR
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    try:
        bundle: DatasetBundle = auto_load(data_dir)
    except FileNotFoundError:
        from .synthetic import write_placeholder_dataset

        write_placeholder_dataset(data_dir)
        bundle = auto_load(data_dir)
    dump_json(
        {
            "source_note": bundle.source_note,
            "label_disclaimer": config.LABEL_DISCLAIMER,
            "emnlp17_lexicon": {
                "upstream_url": config.EMNLP17_UPSTREAM_URL,
                "local_user_selection_dir": str(config.EMNLP17_USER_SELECTION_DIR),
                "files_used": list(config.EMNLP17_LEXICON_FILES),
                "note": "Weak-supervision style cues from EMNLP 2017 user_selection resources; not the paper's Keras model.",
            },
        },
        artifacts_dir / "dataset_meta.json",
    )

    train_df = bundle.train.copy()
    test_df = bundle.test.copy()

    train_df["text_clean"] = train_df["text_raw"].map(preprocess_text_en)
    test_df["text_clean"] = test_df["text_raw"].map(preprocess_text_en)

    y_train_full = _binary_labels(train_df, config.POSITIVE_LABELS)
    y_test = _binary_labels(test_df, config.POSITIVE_LABELS)

    vf = val_fraction if val_fraction is not None else config.VAL_FRACTION
    idx = np.arange(len(train_df))
    tr_idx, va_idx = train_test_split(
        idx,
        test_size=vf,
        random_state=config.RANDOM_STATE,
        stratify=y_train_full,
    )

    tr = train_df.iloc[tr_idx].reset_index(drop=True)
    va = train_df.iloc[va_idx].reset_index(drop=True)
    y_tr = y_train_full[tr_idx]
    y_va = y_train_full[va_idx]

    tr, y_tr = _maybe_subsample(tr, y_tr)

    vectorizer = fit_tfidf(tr["text_clean"])
    scaler = StandardScaler(with_mean=False)

    X_tr = vectorize_text_stats(vectorizer, scaler, tr["text_clean"], fit_scaler=True)
    X_va = vectorize_text_stats(vectorizer, scaler, va["text_clean"], fit_scaler=False)
    X_te = vectorize_text_stats(vectorizer, scaler, test_df["text_clean"], fit_scaler=False)

    if oversample and y_tr.mean() > 0.01 and y_tr.mean() < 0.99:
        ros = RandomOverSampler(random_state=config.RANDOM_STATE)
        X_tr, y_tr = ros.fit_resample(X_tr, y_tr)

    models = _build_models(fast)
    val_results: dict[str, dict[str, float]] = {}
    fitted: dict[str, Any] = {}

    for name, model in models.items():
        model.fit(X_tr, y_tr)
        est = model.best_estimator_ if hasattr(model, "best_estimator_") else model
        if hasattr(est, "predict_proba"):
            p_va = est.predict_proba(X_va)[:, 1]
        else:
            p_va = est.decision_function(X_va)
            p_va = (p_va - p_va.min()) / (p_va.max() - p_va.min() + 1e-9)
        pred_va = (p_va >= 0.5).astype(int)
        val_results[name] = evaluate_binary(y_va, pred_va, p_va)
        fitted[name] = model

    best_name = max(val_results, key=lambda k: val_results[k]["f2"])
    best_model = fitted[best_name]
    best_est = best_model.best_estimator_ if hasattr(best_model, "best_estimator_") else best_model

    if hasattr(best_est, "predict_proba"):
        p_va = best_est.predict_proba(X_va)[:, 1]
        p_te = best_est.predict_proba(X_te)[:, 1]
    else:
        p_va = best_est.decision_function(X_va)
        p_te = best_est.decision_function(X_te)
        p_va = (p_va - p_va.min()) / (p_va.max() - p_va.min() + 1e-9)
        p_te = (p_te - p_te.min()) / (p_te.max() - p_te.min() + 1e-9)

    platt_path = artifacts_dir / "platt_calibrator.pkl"
    platt_note = "none"
    if getattr(config, "USE_PLATT_CALIBRATION", True):
        platt_fit = pcal.fit_platt_calibrator(p_va, y_va)
        if platt_fit is not None:
            p_va_c = platt_fit.predict_proba(np.asarray(p_va, dtype=float).reshape(-1, 1))[:, 1]
            p_te_c = platt_fit.predict_proba(np.asarray(p_te, dtype=float).reshape(-1, 1))[:, 1]
            pcal.save_platt(platt_fit, artifacts_dir)
            p_va, p_te = p_va_c, p_te_c
            platt_note = "platt_validation"
        else:
            platt_path.unlink(missing_ok=True)
    else:
        platt_path.unlink(missing_ok=True)

    thr_f2 = best_threshold_fbeta(y_va, p_va, beta=2.0)
    pred_te = (p_te >= thr_f2).astype(int)

    test_metrics = evaluate_binary(y_test, pred_te, p_te)
    report = classification_report_dict(y_test, pred_te)

    low = float(np.quantile(p_va, 0.33))
    high = float(np.quantile(p_va, 0.66))
    risk = {
        "low": min(config.RISK_LOW_DEFAULT, low),
        "high": max(config.RISK_HIGH_DEFAULT, high),
        "operating_threshold": thr_f2,
        "calibrated_on": "validation_split",
        "probability_calibration": platt_note,
    }

    joblib.dump(best_est, artifacts_dir / "depression_model.pkl")
    joblib.dump(vectorizer, artifacts_dir / "tfidf.pkl")
    joblib.dump(scaler, artifacts_dir / "scaler.pkl")
    (artifacts_dir / "best_model_name.txt").write_text(best_name, encoding="utf-8")
    dump_json(
        {
            "bundle": bundle.source_note,
            "label_disclaimer": config.LABEL_DISCLAIMER,
            "positive_labels": sorted(config.POSITIVE_LABELS),
            "best_model": best_name,
            "val_metrics_per_model": val_results,
            "test_metrics": test_metrics,
            "classification_report": report,
            "risk_thresholds": risk,
        },
        artifacts_dir / "metrics.json",
    )
    dump_json(risk, artifacts_dir / "risk_thresholds.json")

    plot_confusion(y_test, pred_te, artifacts_dir / "confusion_matrix_test.png", title=f"Test confusion ({best_name})")
    plot_roc(y_test, p_te, artifacts_dir / "roc_test.png")
    plot_pr(y_test, p_te, artifacts_dir / "pr_test.png")
    plot_calibration_reliability(y_test, p_te, artifacts_dir / "calibration_reliability_test.png")
    plot_model_bar(val_results, artifacts_dir / "model_comparison_val_f1.png", metric="f1")
    plot_model_bar(val_results, artifacts_dir / "model_comparison_val_f2.png", metric="f2")

    meta_path = artifacts_dir / "dataset_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["best_model"] = best_name
    meta["probability_calibration"] = platt_note
    dump_json(meta, meta_path)

    summary = {
        "best_model": best_name,
        "test_metrics": test_metrics,
        "risk_thresholds": risk,
        "artifacts_dir": str(artifacts_dir),
    }
    return summary


if __name__ == "__main__":
    run_training()
