"""Train reproducible text-classification experiments and write auditable artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from . import config
from . import probability_calibrate as pcal
from .data_db import build_merged_dataset, dataset_stats, load_all_samples, needs_rebuild
from .embedding_baseline import predict_embedding_proba, save_embedding_artifacts, train_embedding_classifier
from .evaluate import (
    best_threshold_fbeta,
    bootstrap_confidence_intervals,
    classification_report_dict,
    derive_risk_thresholds,
    dump_json,
    evaluate_binary,
    plot_calibration_reliability,
    plot_confusion,
    plot_model_bar,
    plot_pr,
    plot_roc,
)
from .features import FEATURE_VARIANTS, build_feature_union
from .io_data import DatasetBundle, auto_load
from .labels import binary_labels
from .oov_eval import run_oov_eval
from .preprocess import preprocess_text_en


def _build_estimators(seed: int, full: bool) -> dict[str, Any]:
    estimators: dict[str, Any] = {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=seed, solver="saga"
        ),
        "linear_svc": CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", dual="auto", max_iter=20000, random_state=seed),
            method="sigmoid",
            cv=3,
        ),
    }
    if full:
        estimators["random_forest"] = RandomForestClassifier(
            n_estimators=250,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
        try:
            from xgboost import XGBClassifier

            estimators["xgboost"] = XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=seed,
                tree_method="hist",
                eval_metric="logloss",
            )
        except Exception:
            pass
    return estimators


def _make_pipeline(estimator: Any, *, oversample: bool, seed: int, variant: str) -> Pipeline:
    steps: list[tuple[str, Any]] = [
        ("features", build_feature_union(variant)),
        ("scale", StandardScaler(with_mean=False)),
    ]
    if oversample:
        steps.append(("oversample", RandomOverSampler(random_state=seed)))
    steps.append(("classifier", estimator))
    return Pipeline(steps)


def _split_validation(
    frame: pd.DataFrame, labels: np.ndarray, seed: int
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    if len(frame) < 20 or np.unique(labels).size < 2:
        return frame, labels, frame, labels
    cal_idx, select_idx = train_test_split(
        np.arange(len(frame)),
        test_size=0.5,
        random_state=seed,
        stratify=labels,
    )
    return (
        frame.iloc[cal_idx].reset_index(drop=True),
        labels[cal_idx],
        frame.iloc[select_idx].reset_index(drop=True),
        labels[select_idx],
    )


def _subsample(frame: pd.DataFrame, labels: np.ndarray, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    cap = int(config.MAX_TRAIN_ROWS)
    if len(frame) <= cap:
        return frame, labels
    idx, _ = train_test_split(
        np.arange(len(frame)), train_size=cap, random_state=seed, stratify=labels
    )
    return frame.iloc[idx].reset_index(drop=True), labels[idx]


def _package_versions() -> dict[str, str]:
    names = ["numpy", "pandas", "scikit-learn", "imbalanced-learn", "xgboost"]
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _git_revision(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=root, text=True
        ).strip()
    except Exception:
        return "unknown"


def _dataset_fingerprint(data_dir: Path) -> str:
    try:
        samples = load_all_samples(data_dir)
        values = sorted(samples["text_fingerprint"].astype(str))
        return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()
    except Exception:
        return "unavailable"


def _per_source_test_metrics(data_dir: Path, y_score: np.ndarray, threshold: float) -> dict[str, Any]:
    try:
        test_rows = load_all_samples(data_dir)
    except Exception:
        return {}
    test_rows = test_rows[test_rows["split_name"] == "test"].reset_index(drop=True)
    if len(test_rows) != len(y_score):
        return {}
    y = binary_labels(test_rows, config.POSITIVE_LABELS)
    output = {}
    for source, idx in test_rows.groupby("source_id").groups.items():
        positions = np.asarray(list(idx), dtype=int)
        scores = y_score[positions]
        output[str(source)] = evaluate_binary(y[positions], (scores >= threshold).astype(int), scores)
    return output


def _write_error_cases(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    path: Path,
) -> None:
    rows = test_df[["text_raw"]].copy()
    rows["label"] = y_true
    rows["model_score"] = y_score
    rows["prediction"] = (y_score >= threshold).astype(int)
    errors = rows[rows["label"] != rows["prediction"]].copy()
    errors["error_type"] = np.where(errors["prediction"] == 1, "false_positive", "false_negative")
    errors["distance_from_threshold"] = np.abs(errors["model_score"] - threshold)
    errors.sort_values("distance_from_threshold", ascending=False).head(100).to_csv(
        path, index=False, encoding="utf-8"
    )


def run_training(
    *,
    data_dir: Path | None = None,
    artifacts_dir: Path | None = None,
    experiment: str = "quick",
    split_strategy: str = "source_label",
    seed: int | None = None,
    fast: bool | None = None,
    oversample: bool = True,
    val_fraction: float | None = None,
    rebuild_data: bool = False,
    auto_rebuild_data: bool | None = None,
    with_embeddings: bool | None = None,
    run_oov: bool | None = None,
) -> dict[str, Any]:
    if experiment not in {"quick", "full"}:
        raise ValueError("experiment must be 'quick' or 'full'")
    if split_strategy not in {"source_label", "label"}:
        raise ValueError("split_strategy must be 'source_label' or 'label'")
    seed = int(config.RANDOM_STATE if seed is None else seed)
    full = experiment == "full"
    if fast is not None:
        full = not fast
        experiment = "full" if full else "quick"

    data_dir = data_dir or config.DATA_DIR
    artifacts_dir = artifacts_dir or config.ARTIFACTS_DIR
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    auto_rebuild = config.AUTO_REBUILD_DATA if auto_rebuild_data is None else auto_rebuild_data
    if rebuild_data or (auto_rebuild and needs_rebuild(data_dir)):
        build_merged_dataset(
            data_dir=data_dir,
            export_csv=True,
            random_state=seed,
            split_strategy=split_strategy,
        )

    try_embeddings = full if with_embeddings is None else with_embeddings
    do_oov = full if run_oov is None else run_oov
    bundle: DatasetBundle = auto_load(data_dir)
    train_df, test_df = bundle.train.copy(), bundle.test.copy()
    train_df["text_clean"] = train_df["text_raw"].map(preprocess_text_en)
    test_df["text_clean"] = test_df["text_raw"].map(preprocess_text_en)
    y_test = binary_labels(test_df, config.POSITIVE_LABELS)

    if bundle.val is not None and len(bundle.val):
        val_df = bundle.val.copy()
        val_df["text_clean"] = val_df["text_raw"].map(preprocess_text_en)
        fit_df = train_df.reset_index(drop=True)
        y_fit = binary_labels(fit_df, config.POSITIVE_LABELS)
        y_val = binary_labels(val_df, config.POSITIVE_LABELS)
    else:
        labels = binary_labels(train_df, config.POSITIVE_LABELS)
        fit_idx, val_idx = train_test_split(
            np.arange(len(train_df)),
            test_size=val_fraction or config.VAL_FRACTION,
            random_state=seed,
            stratify=labels,
        )
        fit_df = train_df.iloc[fit_idx].reset_index(drop=True)
        val_df = train_df.iloc[val_idx].reset_index(drop=True)
        y_fit, y_val = labels[fit_idx], labels[val_idx]

    fit_df, y_fit = _subsample(fit_df, y_fit, seed)
    cal_df, y_cal, select_df, y_select = _split_validation(val_df, y_val, seed)

    oov_report = None
    oov_scores: dict[str, float] = {}
    if do_oov:
        try:
            oov_report = run_oov_eval(
                data_dir=data_dir,
                artifacts_dir=artifacts_dir,
                with_embeddings=try_embeddings,
                random_state=seed,
            )
            for name, report in oov_report.get("tfidf_candidates", {}).items():
                value = report.get("mean_out_of_domain_f2")
                if value is not None:
                    oov_scores[name] = float(value)
        except Exception as exc:
            oov_report = {"error": str(exc)}

    fitted: dict[str, Pipeline] = {}
    validation: dict[str, dict[str, Any]] = {}
    for name, estimator in _build_estimators(seed, full).items():
        pipeline = _make_pipeline(
            clone(estimator),
            oversample=oversample,
            seed=seed,
            variant="tfidf_stats_vader_emnlp",
        )
        pipeline.fit(fit_df["text_clean"], y_fit)
        scores = pipeline.predict_proba(select_df["text_clean"])[:, 1]
        threshold = best_threshold_fbeta(y_select, scores, beta=2.0)
        metrics = evaluate_binary(y_select, (scores >= threshold).astype(int), scores)
        has_required_oov = not oov_scores or name in oov_scores
        combined = float(metrics["f2"])
        if oov_scores and name in oov_scores:
            combined = 0.7 * combined + 0.3 * oov_scores[name]
        validation[name] = {
            "metrics": metrics,
            "threshold": threshold,
            "out_of_domain_f2": oov_scores.get(name),
            "selection_score": combined if has_required_oov else None,
            "selection_eligible": has_required_oov,
        }
        fitted[name] = pipeline

    eligible = [name for name, value in validation.items() if value["selection_eligible"]]
    best_name = max(eligible, key=lambda name: validation[name]["selection_score"])
    best_pipeline = fitted[best_name]
    p_cal_raw = best_pipeline.predict_proba(cal_df["text_clean"])[:, 1]
    calibrator = pcal.fit_platt_calibrator(p_cal_raw, y_cal) if config.USE_PLATT_CALIBRATION else None
    if calibrator is not None:
        pcal.save_platt(calibrator, artifacts_dir)
    else:
        (artifacts_dir / "platt_calibrator.pkl").unlink(missing_ok=True)

    p_select = best_pipeline.predict_proba(select_df["text_clean"])[:, 1]
    p_test = best_pipeline.predict_proba(test_df["text_clean"])[:, 1]
    if calibrator is not None:
        p_select = calibrator.predict_proba(p_select.reshape(-1, 1))[:, 1]
        p_test = calibrator.predict_proba(p_test.reshape(-1, 1))[:, 1]
    threshold = best_threshold_fbeta(y_select, p_select, beta=2.0)
    pred_test = (p_test >= threshold).astype(int)
    test_metrics = evaluate_binary(y_test, pred_test, p_test)
    intervals = bootstrap_confidence_intervals(y_test, pred_test, p_test, random_state=seed)
    risk = {
        **derive_risk_thresholds(
            y_select,
            p_select,
            low_default=config.RISK_LOW_DEFAULT,
            high_default=config.RISK_HIGH_DEFAULT,
        ),
        "operating_threshold": threshold,
        "calibrated_on": "validation_calibration_half",
        "threshold_selected_on": "validation_selection_half",
        "probability_calibration": "platt_separate_validation_half" if calibrator else "none",
    }

    ablation: dict[str, Any] = {}
    if full:
        for variant in FEATURE_VARIANTS:
            pipeline = _make_pipeline(
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=seed,
                    solver="saga",
                ),
                oversample=oversample,
                seed=seed,
                variant=variant,
            )
            pipeline.fit(fit_df["text_clean"], y_fit)
            scores = pipeline.predict_proba(select_df["text_clean"])[:, 1]
            variant_threshold = best_threshold_fbeta(y_select, scores, beta=2.0)
            ablation[variant] = evaluate_binary(
                y_select, (scores >= variant_threshold).astype(int), scores
            )
        pd.DataFrame(ablation).T.to_csv(artifacts_dir / "ablation_results.csv", encoding="utf-8")
    else:
        (artifacts_dir / "ablation_results.csv").unlink(missing_ok=True)

    embedding_summary = None
    if try_embeddings:
        try:
            emb_clf, emb_scaler, emb_val, emb_threshold = train_embedding_classifier(
                fit_df["text_clean"],
                y_fit,
                select_df["text_clean"],
                y_select,
                random_state=seed,
            )
            save_embedding_artifacts(
                emb_clf,
                emb_scaler,
                model_name=config.EMBEDDING_MODEL_NAME,
                threshold=emb_threshold,
                val_metrics=emb_val,
                artifacts_dir=artifacts_dir,
            )
            emb_test_scores = predict_embedding_proba(test_df["text_clean"], emb_clf, emb_scaler)
            embedding_summary = {
                "validation": emb_val,
                "in_domain_test": evaluate_binary(
                    y_test, (emb_test_scores >= emb_threshold).astype(int), emb_test_scores
                ),
                "threshold": emb_threshold,
            }
        except (ImportError, OSError) as exc:
            embedding_summary = {"status": "skipped", "reason": str(exc)}
    else:
        for name in (
            "embedding_classifier.pkl",
            "embedding_scaler.pkl",
            "embedding_model_name.txt",
            "embedding_metrics.json",
        ):
            (artifacts_dir / name).unlink(missing_ok=True)

    trained_at = datetime.now(timezone.utc).isoformat()
    root = Path(__file__).resolve().parents[2]
    git_revision = _git_revision(root)
    model_version = f"{trained_at[:10]}-{git_revision}"
    metadata = {
        "model_version": model_version,
        "trained_at": trained_at,
        "git_revision": git_revision,
        "experiment": experiment,
        "split_strategy": split_strategy,
        "random_seed": seed,
        "dataset_fingerprint": _dataset_fingerprint(data_dir),
        "dependencies": _package_versions(),
        "source_note": bundle.source_note,
        "dataset_build": dataset_stats(data_dir),
        "label_disclaimer": config.LABEL_DISCLAIMER,
        "score_disclaimer": "Model-positive score, not a clinical probability or diagnosis.",
        "best_model": best_name,
    }

    joblib.dump(best_pipeline, artifacts_dir / "model_pipeline.pkl")
    feature_union = best_pipeline.named_steps["features"]
    vectorizer = dict(feature_union.transformer_list)["tfidf"]
    joblib.dump(best_pipeline.named_steps["classifier"], artifacts_dir / "depression_model.pkl")
    joblib.dump(vectorizer, artifacts_dir / "tfidf.pkl")
    joblib.dump(best_pipeline.named_steps["scale"], artifacts_dir / "scaler.pkl")
    (artifacts_dir / "best_model_name.txt").write_text(best_name, encoding="utf-8")
    dump_json(metadata, artifacts_dir / "model_metadata.json")
    dump_json(risk, artifacts_dir / "risk_thresholds.json")

    metrics_payload = {
        "model_version": model_version,
        "validation": {
            "protocol": "calibration and threshold/model-selection use disjoint validation halves",
            "models": validation,
            "ablation": ablation,
        },
        "in_domain_test": {
            "metrics": test_metrics,
            "confidence_intervals_95": intervals,
            "classification_report": classification_report_dict(y_test, pred_test),
            "per_source": _per_source_test_metrics(data_dir, p_test, threshold),
        },
        "out_of_domain_test": oov_report or {"status": "not_run"},
        "calibration": {
            "method": risk["probability_calibration"],
            "brier_score": test_metrics.get("brier"),
        },
        "risk_thresholds": risk,
        "embedding_baseline": embedding_summary or {"status": "not_run"},
        "label_disclaimer": config.LABEL_DISCLAIMER,
    }
    dump_json(metrics_payload, artifacts_dir / "metrics.json")
    pd.DataFrame(
        {
            name: {
                **value["metrics"],
                "threshold": value["threshold"],
                "out_of_domain_f2": value["out_of_domain_f2"],
                "selection_score": value["selection_score"],
            }
            for name, value in validation.items()
        }
    ).T.to_csv(artifacts_dir / "experiment_results.csv", encoding="utf-8")
    dump_json(metadata, artifacts_dir / "dataset_meta.json")
    _write_error_cases(
        test_df, y_test, p_test, threshold, artifacts_dir / "error_cases_test.csv"
    )

    plot_confusion(
        y_test,
        pred_test,
        artifacts_dir / "confusion_matrix_test.png",
        title=f"Test confusion ({best_name})",
    )
    plot_roc(y_test, p_test, artifacts_dir / "roc_test.png")
    plot_pr(y_test, p_test, artifacts_dir / "pr_test.png")
    plot_calibration_reliability(
        y_test, p_test, artifacts_dir / "calibration_reliability_test.png"
    )
    plot_model_bar(
        {name: value["metrics"] for name, value in validation.items()},
        artifacts_dir / "model_comparison_val_f2.png",
        metric="f2",
    )
    (artifacts_dir / "model_comparison_val_f1.png").unlink(missing_ok=True)

    return {
        "model_version": model_version,
        "best_model": best_name,
        "in_domain_test": test_metrics,
        "risk_thresholds": risk,
        "artifacts_dir": str(artifacts_dir),
    }


if __name__ == "__main__":
    run_training()
