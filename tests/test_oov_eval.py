"""Tests for OOV evaluation helpers and auto-rebuild detection."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def test_load_all_samples_includes_source_id(tmp_path: Path):
    from depression_ml.data_db import build_dataset, load_all_samples

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rows = 80
    src = data_dir / "source.csv"
    df = pd.DataFrame(
        {
            "clean_text": [
                (
                    f"post type {'depression' if i % 2 else 'normal'} marker {chr(97 + i % 26)} "
                    f"extra words about feelings support and daily experience topic {chr(122 - i % 13)}"
                )
                for i in range(rows)
            ],
            "is_depression": [0] * (rows // 2) + [1] * (rows - rows // 2),
        }
    )
    df.to_csv(src, index=False)
    build_dataset(src, data_dir=data_dir, export_csv=False)
    all_df = load_all_samples(data_dir)
    assert "source_id" in all_df.columns
    assert len(all_df) >= 20


def test_needs_rebuild_when_db_missing(tmp_path: Path):
    from depression_ml.data_db import needs_rebuild

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    assert needs_rebuild(data_dir) is True
