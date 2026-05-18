"""Tests for multi-source CSV import."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def test_sentiment_loader_binary(tmp_path: Path):
    from depression_ml.data_import import SourceSpec, load_source_frame

    p = tmp_path / "combined_data.csv"
    pd.DataFrame(
        {
            "statement": [
                "I feel hopeless and cannot sleep",
                "Had a great day at the park with friends",
            ],
            "status": ["Depression", "Normal"],
        }
    ).to_csv(p, index=False)

    spec = SourceSpec(
        id="sentiment",
        file="combined_data.csv",
        text_column="statement",
        label_column="status",
        label_type="mental_health_status",
    )
    frame = load_source_frame(p, spec, english_only=False, min_chars=5)
    assert len(frame) == 2
    assert set(frame["label_raw"]) == {"0", "1"}


def test_urdu_loader_english_and_severity(tmp_path: Path):
    from depression_ml.data_import import SourceSpec, load_source_frame

    p = tmp_path / "urdu_depression_dataset.csv"
    pd.DataFrame(
        {
            "text_english": [
                "I have been feeling severe depression lately and need help",
                "Today was a normal productive day at work",
            ],
            "text_roman": ["", ""],
            "depression_label": [3, 0],
        }
    ).to_csv(p, index=False)

    spec = SourceSpec(
        id="urdu",
        file="urdu_depression_dataset.csv",
        text_columns=("text_english", "text_roman"),
        label_column="depression_label",
        label_type="urdu_severity",
    )
    frame = load_source_frame(p, spec, english_only=True, min_chars=10)
    assert len(frame) == 2
    assert frame.iloc[0]["label_raw"] == "1"
    assert frame.iloc[1]["label_raw"] == "0"


def test_merge_manifest_reddit_only(tmp_path: Path):
    from depression_ml.data_import import discover_and_load

    manifest = {
        "sources": [
            {
                "id": "reddit",
                "file": "tiny_reddit.csv",
                "text_column": "clean_text",
                "label_column": "is_depression",
                "label_type": "binary_numeric",
                "required": True,
            }
        ]
    }
    (tmp_path / "sources.json").write_text(__import__("json").dumps(manifest), encoding="utf-8")
    rows = 60
    pd.DataFrame(
        {
            "clean_text": [
                f"post type {'depression' if i % 2 else 'normal'} marker {chr(97 + i % 26)} "
                f"extra words about feelings support and daily experience"
                for i in range(rows)
            ],
            "is_depression": [0] * (rows // 2) + [1] * (rows - rows // 2),
        }
    ).to_csv(tmp_path / "tiny_reddit.csv", index=False)

    merged, report = discover_and_load(tmp_path, source_ids=["reddit"], english_only=False, min_chars=10)
    assert len(merged) >= 20
    assert report["loaded"][0]["id"] == "reddit"
