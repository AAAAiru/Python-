"""Feature ablation + split-protocol comparison for depression screening."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _log(msg: str) -> None:
    print(msg, flush=True)

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from . import config
from .evaluate import best_threshold_fbeta, dump_json, evaluate_binary, plot_model_bar
from .features_ablation import FEATURE_SET_NAMES, fit_vectorizers, vectorize_feature_set
from .io_data import DatasetBundle, auto_load
from .labels import binary_labels
from .preprocess import preprocess_text_en
from .splits import SplitProtocol, bundle_from_protocol, count_groups


def _prepare_bundle(bundle: DatasetBundle) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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


def _subsample_train(tr: pd.DataFrame, y_tr: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
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


def _fit_eval_lr(
    X_tr,
    y_tr: np.ndarray,
    X_va,
    y_va: np.ndarray,
    X_te,
    y_te: np.ndarray,
    *,
    oversample: bool,
) -> dict[str, Any]:
    if oversample and 0.01 < y_tr.mean() < 0.99:
        ros = RandomOverSampler(random_state=config.RANDOM_STATE)
        X_tr, y_tr = ros.fit_resample(X_tr, y_tr)

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="saga",
        random_state=config.RANDOM_STATE,
    )
    clf.fit(X_tr, y_tr)
    p_va = clf.predict_proba(X_va)[:, 1]
    p_te = clf.predict_proba(X_te)[:, 1]
    thr = best_threshold_fbeta(y_va, p_va, beta=2.0)
    pred_va = (p_va >= thr).astype(int)
    pred_te = (p_te >= thr).astype(int)
    return {
        "val": evaluate_binary(y_va, pred_va, p_va),
        "test": evaluate_binary(y_te, pred_te, p_te),
        "threshold_f2": float(thr),
    }


def _cv_f2(
    X,
    y: np.ndarray,
    groups: np.ndarray | None,
    *,
    n_folds: int,
) -> dict[str, float]:
    n_folds = int(min(n_folds, max(2, len(y) // 50)))
    if groups is not None and len(np.unique(groups)) >= n_folds:
        splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=config.RANDOM_STATE)
        split_iter = splitter.split(X, y, groups=groups)
    else:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.RANDOM_STATE)
        split_iter = splitter.split(X, y)

    fold_metrics: list[dict[str, float]] = []
    for tr_idx, va_idx in split_iter:
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        if 0.01 < y_tr.mean() < 0.99:
            ros = RandomOverSampler(random_state=config.RANDOM_STATE)
            X_tr, y_tr = ros.fit_resample(X_tr, y_tr)
        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="saga",
            random_state=config.RANDOM_STATE,
        )
        clf.fit(X_tr, y_tr)
        p_va = clf.predict_proba(X_va)[:, 1]
        thr = best_threshold_fbeta(y_va, p_va, beta=2.0)
        pred_va = (p_va >= thr).astype(int)
        fold_metrics.append(evaluate_binary(y_va, pred_va, p_va))

    keys = fold_metrics[0].keys()
    return {k: float(np.mean([m[k] for m in fold_metrics])) for k in keys}


def run_one_feature_set(
    feature_set: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    oversample: bool = True,
    cv_folds: int = 5,
    groups_for_cv: np.ndarray | None = None,
) -> dict[str, Any]:
    tr, va, te = train_df, val_df, test_df
    y_tr = binary_labels(tr, config.POSITIVE_LABELS)
    y_va = binary_labels(va, config.POSITIVE_LABELS)
    y_te = binary_labels(te, config.POSITIVE_LABELS)
    tr, y_tr = _subsample_train(tr, y_tr)

    vectorizer = fit_vectorizers(tr["text_clean"])
    scaler = StandardScaler(with_mean=False)
    X_tr = vectorize_feature_set(feature_set, vectorizer, scaler, tr["text_clean"], fit_scaler=True)
    X_va = vectorize_feature_set(feature_set, vectorizer, scaler, va["text_clean"], fit_scaler=False)
    X_te = vectorize_feature_set(feature_set, vectorizer, scaler, te["text_clean"], fit_scaler=False)

    out = _fit_eval_lr(X_tr, y_tr, X_va, y_va, X_te, y_te, oversample=oversample)
    out["feature_set"] = feature_set

    cv_df = pd.concat([tr, va], ignore_index=True)
    y_cv = binary_labels(cv_df, config.POSITIVE_LABELS)
    cv_df, y_cv = _subsample_train(cv_df, y_cv) if len(cv_df) > config.MAX_TRAIN_ROWS else (cv_df, y_cv)
    vec_cv = fit_vectorizers(cv_df["text_clean"])
    sc_cv = StandardScaler(with_mean=False)
    X_cv = vectorize_feature_set(feature_set, vec_cv, sc_cv, cv_df["text_clean"], fit_scaler=True)
    g_cv = groups_for_cv
    if g_cv is not None and len(g_cv) != len(cv_df):
        g_cv = None
    out["cv_train_val_f2"] = _cv_f2(X_cv, y_cv, g_cv, n_folds=cv_folds)
    return out


def run_feature_ablation(
    bundle: DatasetBundle,
    *,
    feature_sets: tuple[str, ...] | None = None,
    oversample: bool = True,
    cv_folds: int | None = None,
    groups_for_cv: np.ndarray | None = None,
) -> dict[str, dict[str, Any]]:
    feature_sets = feature_sets or FEATURE_SET_NAMES
    folds = int(cv_folds if cv_folds is not None else getattr(config, "ABLATION_CV_FOLDS", 5))
    train_df, val_df, test_df = _prepare_bundle(bundle)
    results: dict[str, dict[str, Any]] = {}
    for i, fs in enumerate(feature_sets, start=1):
        _log(f"[ablation] feature set {i}/{len(feature_sets)}: {fs} (fit + {cv_folds}-fold CV, may take several minutes)...")
        results[fs] = run_one_feature_set(
            fs,
            train_df,
            val_df,
            test_df,
            oversample=oversample,
            cv_folds=folds,
            groups_for_cv=groups_for_cv,
        )
        _log(f"[ablation]   done {fs}: test F2={results[fs]['test']['f2']:.4f}")
    return results


def run_split_protocol_compare(
    merged: pd.DataFrame,
    *,
    feature_set: str = "tfidf_full",
    oversample: bool = True,
    cv_folds: int | None = None,
) -> dict[str, dict[str, Any]] | None:
    """Compare stratified row split vs group-by-source on the same merged pool."""
    if count_groups(merged) < 2:
        return None
    folds = int(cv_folds if cv_folds is not None else getattr(config, "ABLATION_CV_FOLDS", 5))
    out: dict[str, dict[str, Any]] = {}
    for i, protocol in enumerate(("stratified_row", "group_by_source"), start=1):
        _log(f"[ablation] split protocol {i}/2: {protocol}...")
        bundle = bundle_from_protocol(merged, protocol)
        train_df, val_df, test_df = _prepare_bundle(bundle)
        groups_cv = None
        if "source_id" in train_df.columns:
            cv_part = pd.concat([train_df, val_df], ignore_index=True)
            groups_cv = cv_part["source_id"].astype(str).to_numpy()
        row = run_one_feature_set(
            feature_set,
            train_df,
            val_df,
            test_df,
            oversample=oversample,
            cv_folds=folds,
            groups_for_cv=groups_cv,
        )
        row["split_protocol"] = protocol
        row["n_sources"] = count_groups(merged)
        out[protocol] = row
    return out


def _results_to_table(feature_ablation: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for name, res in feature_ablation.items():
        rows.append(
            {
                "feature_set": name,
                "val_f1": res["val"]["f1"],
                "val_f2": res["val"]["f2"],
                "val_recall": res["val"]["recall"],
                "test_f1": res["test"]["f1"],
                "test_f2": res["test"]["f2"],
                "test_recall": res["test"]["recall"],
                "test_roc_auc": res["test"].get("roc_auc"),
                "cv_f2": res["cv_train_val_f2"]["f2"],
            }
        )
    return pd.DataFrame(rows)


def run_ablation(
    *,
    data_dir: Path | None = None,
    artifacts_dir: Path | None = None,
    oversample: bool = True,
    cv_folds: int | None = None,
    compare_splits: bool = True,
) -> dict[str, Any]:
    data_dir = data_dir or config.DATA_DIR
    artifacts_dir = artifacts_dir or config.ARTIFACTS_DIR
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    folds = int(cv_folds if cv_folds is not None else getattr(config, "ABLATION_CV_FOLDS", 5))

    _log("[ablation] loading dataset...")
    bundle = auto_load(data_dir)
    _log(f"[ablation] {bundle.source_note}")
    _log(f"[ablation] feature ablation: {list(FEATURE_SET_NAMES)}, cv_folds={folds}")
    feature_ablation = run_feature_ablation(bundle, oversample=oversample, cv_folds=folds)

    split_compare = None
    merged = None
    try:
        from .data_db import load_all_samples

        merged = load_all_samples(data_dir)
        merged["text_clean"] = merged["text_raw"].map(preprocess_text_en)
        if compare_splits:
            _log("[ablation] split-protocol compare (2 extra full runs on tfidf_full)...")
            try:
                split_compare = run_split_protocol_compare(merged, oversample=oversample, cv_folds=folds)
            except Exception as exc:
                split_compare = {"error": str(exc)}
    except FileNotFoundError:
        pass

    report: dict[str, Any] = {
        "positive_labels": sorted(config.POSITIVE_LABELS),
        "data_note": bundle.source_note,
        "feature_ablation": feature_ablation,
        "split_protocol_compare": split_compare,
        "notes": {
            "feature_sets": list(FEATURE_SET_NAMES),
            "cv_folds": folds,
            "group_split_by_source": split_compare is not None,
            "classifier": "logistic_regression_saga_balanced",
            "max_train_rows": config.MAX_TRAIN_ROWS,
        },
    }

    _log("[ablation] writing artifacts...")
    dump_json(report, artifacts_dir / "ablation_report.json")
    table = _results_to_table(feature_ablation)
    table.to_csv(artifacts_dir / "ablation_table.csv", index=False, encoding="utf-8")

    bar_vals = {k: {"f2": v["test"]["f2"]} for k, v in feature_ablation.items()}
    plot_model_bar(bar_vals, artifacts_dir / "ablation_test_f2.png", metric="f2")

    if split_compare and "error" not in split_compare:
        split_bar = {k: {"f2": v["test"]["f2"]} for k, v in split_compare.items()}
        plot_model_bar(split_bar, artifacts_dir / "ablation_split_protocol_f2.png", metric="f2")

    return report
