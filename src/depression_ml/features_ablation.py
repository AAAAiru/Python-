"""Feature-matrix builders for ablation (subset of dense stats + TF-IDF)."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.sparse import csr_matrix, hstack as sparse_hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from . import config
from .features import STAT_FEATURE_ORDER, extract_stat_features, fit_tfidf

# Keys used in ablation reports.
FEATURE_SET_NAMES: tuple[str, ...] = ("tfidf_only", "tfidf_vader", "tfidf_full")

FEATURE_SET_EXTRA_COLS: dict[str, tuple[str, ...]] = {
    "tfidf_only": (),
    "tfidf_vader": ("sentiment_compound",),
    "tfidf_full": STAT_FEATURE_ORDER,
}


def _stats_matrix(texts: Iterable[str], extra_cols: tuple[str, ...]) -> csr_matrix:
    text_list = list(texts)
    if not extra_cols:
        return csr_matrix((len(text_list), 0))
    rows = [[extract_stat_features(t)[k] for k in extra_cols] for t in text_list]
    return csr_matrix(np.asarray(rows, dtype=float))


def vectorize_feature_set(
    feature_set: str,
    vectorizer: TfidfVectorizer,
    scaler: StandardScaler,
    texts: Iterable[str],
    *,
    fit_scaler: bool,
) -> csr_matrix:
    if feature_set not in FEATURE_SET_EXTRA_COLS:
        raise ValueError(f"Unknown feature_set {feature_set!r}; choose from {FEATURE_SET_NAMES}")
    text_list = list(texts)
    X_txt = vectorizer.transform(text_list)
    stats_sp = _stats_matrix(text_list, FEATURE_SET_EXTRA_COLS[feature_set])
    if stats_sp.shape[1] == 0:
        X = X_txt
    else:
        X = sparse_hstack([X_txt, stats_sp], format="csr")
    if fit_scaler:
        return scaler.fit_transform(X)
    return scaler.transform(X)


def fit_vectorizers(train_texts: Iterable[str]) -> TfidfVectorizer:
    return fit_tfidf(train_texts)
