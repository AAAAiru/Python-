"""Lightweight regression tests for data loading, labels, and lexicon."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def test_binary_labels_numeric():
    from depression_ml.labels import binary_labels

    df = pd.DataFrame({"label_raw": [0, 1, 0, 1]})
    y = binary_labels(df, {"Depression"})
    assert np.array_equal(y, np.array([0, 1, 0, 1]))


def test_binary_labels_string_class():
    from depression_ml.labels import binary_labels

    df = pd.DataFrame({"label_raw": ["Normal", "Depression", "Normal"]})
    y = binary_labels(df, {"Depression"})
    assert np.array_equal(y, np.array([0, 1, 0]))


def test_auto_load_reddit_cleaned(tmp_path: Path):
    from depression_ml.io_data import auto_load

    rows = 40
    df = pd.DataFrame(
        {
            "clean_text": [f"sample post number {i} about mood and support" for i in range(rows)],
            "is_depression": [0] * (rows // 2) + [1] * (rows - rows // 2),
        }
    )
    p = tmp_path / "depression_dataset_reddit_cleaned.csv"
    df.to_csv(p, index=False)
    bundle = auto_load(tmp_path)
    assert "depression_dataset_reddit_cleaned" in bundle.source_note
    assert len(bundle.train) > 0 and len(bundle.test) > 0


def test_emnlp17_counts_nonempty():
    from depression_ml.emnlp17_signals import extract_emnlp17_features

    text = "i have major depression and talked to my psychiatrist yesterday"
    feats = extract_emnlp17_features(text)
    assert feats["emnlp_mh_hits"] >= 0.0
    assert feats["emnlp_pos_diag"] >= 0.0
    assert "emnlp_subreddit_word_hits" in feats
    assert "emnlp_subreddit_r_hits" in feats


def test_emnlp17_subreddit_r_style():
    from depression_ml.preprocess import preprocess_text_en
    from depression_ml.emnlp17_signals import extract_emnlp17_features

    raw = "crossposted from r/bpd and r/suicidewatch please read"
    clean = preprocess_text_en(raw)
    feats = extract_emnlp17_features(clean)
    assert feats["emnlp_subreddit_r_hits"] >= 1.0


def test_build_dataset_sqlite(tmp_path: Path):
    from depression_ml.data_db import build_dataset, load_from_db

    src = tmp_path / "source.csv"
    rows = 80
    df = pd.DataFrame(
        {
            "clean_text": [
                (
                    f"post type {'depression' if i % 2 else 'normal'} marker {chr(97 + i % 26)} "
                    f"extra words about feelings support and daily experience"
                )
                for i in range(rows)
            ],
            "is_depression": [0] * (rows // 2) + [1] * (rows - rows // 2),
        }
    )
    df.to_csv(src, index=False)
    stats = build_dataset(src, data_dir=tmp_path, min_chars=10, export_csv=True)
    assert stats["rows_after_clean"] > 0
    bundle = load_from_db(tmp_path)
    assert bundle.val is not None and len(bundle.val) > 0
    assert len(bundle.train) + len(bundle.val) + len(bundle.test) == stats["rows_after_clean"]
    from depression_ml.probability_calibrate import apply_platt, fit_platt_calibrator

    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, size=200)
    y = (rng.uniform(size=200) < p).astype(int)
    cal = fit_platt_calibrator(p, y)
    assert cal is not None
    out = apply_platt(cal, 0.5)
    assert 0.0 < out < 1.0
