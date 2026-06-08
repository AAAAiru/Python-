"""Regression tests for leakage prevention and reproducible model artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def _write_dataset(path: Path, rows: int = 120) -> None:
    def marker(value: int) -> str:
        return "".join(
            chr(97 + digit)
            for digit in (value // (26 * 26), (value // 26) % 26, value % 26)
        )

    half = rows // 2
    texts = [
        f"normal daily routine friends study sunshine unique sample {marker(i)}"
        for i in range(half)
    ] + [
        f"depression hopeless empty exhausted sleep unique sample {marker(i)}"
        for i in range(half, rows)
    ]
    pd.DataFrame({"clean_text": texts, "is_depression": [0] * half + [1] * (rows - half)}).to_csv(
        path, index=False
    )


def test_split_fingerprints_do_not_overlap(tmp_path: Path):
    from depression_ml.data_db import build_dataset, load_all_samples

    source = tmp_path / "source.csv"
    _write_dataset(source)
    build_dataset(source, data_dir=tmp_path, min_chars=10, export_csv=False)
    samples = load_all_samples(tmp_path)
    split_sets = {
        split: set(part["text_fingerprint"])
        for split, part in samples.groupby("split_name")
    }
    assert split_sets["train"].isdisjoint(split_sets["val"])
    assert split_sets["train"].isdisjoint(split_sets["test"])
    assert split_sets["val"].isdisjoint(split_sets["test"])


def test_pipeline_roundtrip_scores_match(tmp_path: Path):
    from depression_ml.train import _make_pipeline
    from sklearn.linear_model import LogisticRegression

    texts = pd.Series(
        [f"calm ordinary day sample {i}" for i in range(20)]
        + [f"hopeless depressed empty sample {i}" for i in range(20)]
    )
    labels = np.array([0] * 20 + [1] * 20)
    pipeline = _make_pipeline(
        LogisticRegression(max_iter=1000),
        oversample=False,
        seed=42,
        variant="tfidf_stats_vader_emnlp",
    )
    pipeline.fit(texts, labels)
    before = pipeline.predict_proba(texts.iloc[:4])[:, 1]
    path = tmp_path / "pipeline.pkl"
    joblib.dump(pipeline, path)
    after = joblib.load(path).predict_proba(texts.iloc[:4])[:, 1]
    assert np.allclose(before, after)


def test_quick_training_writes_consistent_thresholds(tmp_path: Path):
    from depression_ml.data_db import build_dataset
    from depression_ml.train import run_training

    data_dir = tmp_path / "data"
    artifacts_dir = tmp_path / "artifacts"
    data_dir.mkdir()
    source = data_dir / "source.csv"
    _write_dataset(source)
    build_dataset(source, data_dir=data_dir, min_chars=10, export_csv=False)
    run_training(
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
        experiment="quick",
        auto_rebuild_data=False,
        with_embeddings=False,
        run_oov=False,
    )
    metrics = json.loads((artifacts_dir / "metrics.json").read_text(encoding="utf-8"))
    thresholds = json.loads(
        (artifacts_dir / "risk_thresholds.json").read_text(encoding="utf-8")
    )
    assert metrics["risk_thresholds"] == thresholds
    assert (artifacts_dir / "model_pipeline.pkl").exists()
    assert metrics["in_domain_test"]["confidence_intervals_95"]
