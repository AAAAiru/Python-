"""Feature builders: TF-IDF, simple stats, VADER sentiment."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack as sparse_hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from . import config


_vader = SentimentIntensityAnalyzer()


def compound_sentiment(text: str) -> float:
    if not text:
        return 0.0
    return float(_vader.polarity_scores(text)["compound"])


def extract_stat_features(text: str) -> dict[str, float]:
    words = text.split()
    char_len = float(len(text))
    word_count = float(len(words))
    avg_word_len = float(np.mean([len(w) for w in words])) if words else 0.0
    return {
        "char_len": char_len,
        "word_count": word_count,
        "avg_word_len": avg_word_len,
        "sentiment_compound": compound_sentiment(text),
    }


def fit_tfidf(corpus: Iterable[str]) -> TfidfVectorizer:
    vec = TfidfVectorizer(
        max_features=config.TFIDF_MAX_FEATURES,
        ngram_range=config.TFIDF_NGRAM_RANGE,
        min_df=config.TFIDF_MIN_DF,
        max_df=config.TFIDF_MAX_DF,
        sublinear_tf=True,
    )
    vec.fit(corpus)
    return vec


def _stats_matrix(texts: Iterable[str]) -> csr_matrix:
    rows = [[*extract_stat_features(t).values()] for t in texts]
    arr = np.asarray(rows, dtype=float)
    return csr_matrix(arr)


def vectorize_text_stats(
    vectorizer: TfidfVectorizer,
    scaler: StandardScaler,
    texts: Iterable[str],
    *,
    fit_scaler: bool,
) -> csr_matrix:
    X_txt = vectorizer.transform(texts)
    stats_sp = _stats_matrix(texts)
    X = sparse_hstack([X_txt, stats_sp], format="csr")
    if fit_scaler:
        return scaler.fit_transform(X)
    return scaler.transform(X)


def vectorize_single(vectorizer: TfidfVectorizer, scaler: StandardScaler, text: str) -> csr_matrix:
    return vectorize_text_stats(vectorizer, scaler, [text], fit_scaler=False)
