"""Metrics, plots, and threshold search."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def f2_score(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=2, zero_division=0)


def evaluate_binary(y_true, y_pred, y_score=None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2": float(f2_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }
    if y_score is not None:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        prec, rec, thr = precision_recall_curve(y_true, y_score)
        out["pr_auc"] = float(auc(rec, prec))
    return out


def plot_confusion(y_true, y_pred, out_path: Path, title: str = "Confusion matrix") -> None:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Depression"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_roc(y_true, y_score, out_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_pr(y_true, y_score, out_path: Path) -> None:
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    pr_a = auc(rec, prec)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(rec, prec, label=f"PR (AUC={pr_a:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–recall curve")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_calibration_reliability(
    y_true,
    y_score,
    out_path: Path,
    title: str = "Reliability (test set)",
) -> None:
    """Fraction of positives vs mean predicted score (uniform bins)."""
    from sklearn.calibration import calibration_curve

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    n_bins = int(min(10, max(3, len(y_true) // 80)))
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=n_bins, strategy="uniform")
    except ValueError:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(prob_pred, prob_true, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Ideal")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_model_bar(results: dict[str, dict[str, float]], out_path: Path, metric: str = "f1") -> None:
    names = list(results.keys())
    vals = [results[n][metric] for n in names]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(names))
    ax.bar(x, vals, color="#4C72B0")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Model comparison ({metric})")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def best_threshold_fbeta(y_true, y_score, beta: float = 2.0) -> float:
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    # precision_recall_curve last precision/recall are edge values; thr has len-1
    best_t = 0.5
    best = -1.0
    for i in range(len(thr)):
        p, r = prec[i], rec[i]
        if p + r == 0:
            f = 0.0
        else:
            b2 = beta * beta
            f = (1 + b2) * (p * r) / (b2 * p + r + 1e-12)
        if f > best:
            best = f
            best_t = float(thr[i])
    return best_t


def derive_risk_thresholds(
    y_true,
    y_score,
    *,
    low_default: float = 0.35,
    high_default: float = 0.70,
    neg_quantile: float = 0.92,
    pos_quantile: float = 0.12,
) -> dict[str, float]:
    """Derive low/medium/high cutoffs from validation scores (class-conditional quantiles).

    ``low``: upper bound for 低风险 — most validation negatives fall below this.
    ``high``: lower bound for 高风险 — most validation positives fall above this.
    """
    y = np.asarray(y_true)
    p = np.asarray(y_score, dtype=float)
    neg = p[y == 0]
    pos = p[y == 1]

    thr_f2 = best_threshold_fbeta(y, p, beta=2.0)

    if len(neg) >= 10:
        low = float(np.quantile(neg, neg_quantile))
        low = float(np.clip(low, 0.28, low_default))
    else:
        low = float(low_default)

    if len(pos) >= 10:
        high = float(np.quantile(pos, pos_quantile))
        high = float(np.clip(high, high_default, 0.88))
    else:
        high = float(high_default)

    if low >= high - 0.08:
        low, high = float(low_default), float(high_default)

    return {
        "low": low,
        "high": high,
        "operating_threshold": thr_f2,
        "derived_from": "class_conditional_quantiles",
    }


def dump_json(obj: Any, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def classification_report_dict(y_true, y_pred) -> dict[str, Any]:
    return classification_report(y_true, y_pred, target_names=["Normal", "Depression"], output_dict=True, zero_division=0)
