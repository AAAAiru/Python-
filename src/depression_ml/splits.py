"""Train/val/test splitting: row-stratified vs group-by-source."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config
from .io_data import DatasetBundle


SplitProtocol = Literal["stratified_row", "group_by_source"]

MIN_SOURCES_FOR_GROUP_SPLIT = 3


def _label_series(df: pd.DataFrame) -> pd.Series:
    return df["label_raw"].astype(str)


def _source_level_labels(df: pd.DataFrame, group_col: str) -> pd.Series:
    """One pseudo-label per source: majority class among its rows."""
    def _majority(s: pd.Series) -> str:
        m = s.mode()
        return str(m.iloc[0]) if len(m) else str(s.iloc[0])

    return df.groupby(group_col, observed=True)["label_raw"].apply(_majority)


def stratified_row_splits(
    df: pd.DataFrame,
    *,
    train_ratio: float | None = None,
    val_ratio: float | None = None,
    test_ratio: float | None = None,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Same logic as data_db._stratified_splits but returns three frames."""
    tr = float(train_ratio if train_ratio is not None else config.DATASET_TRAIN_RATIO)
    vr = float(val_ratio if val_ratio is not None else config.DATASET_VAL_RATIO)
    te = float(test_ratio if test_ratio is not None else config.DATASET_TEST_RATIO)
    rs = int(random_state if random_state is not None else config.RANDOM_STATE)
    y = _label_series(df)
    tr_val, test_df = train_test_split(df, test_size=te, random_state=rs, stratify=y)
    val_rel = vr / (tr + vr)
    train_df, val_df = train_test_split(
        tr_val,
        test_size=val_rel,
        random_state=rs,
        stratify=_label_series(tr_val),
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def group_by_source_splits(
    df: pd.DataFrame,
    *,
    train_ratio: float | None = None,
    val_ratio: float | None = None,
    test_ratio: float | None = None,
    random_state: int | None = None,
    group_col: str = "source_id",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Assign whole sources to train/val/test (reduces same-corpus leakage)."""
    if group_col not in df.columns:
        raise ValueError(f"Missing {group_col!r} for group split")

    n_sources = int(df[group_col].astype(str).nunique())
    if n_sources < MIN_SOURCES_FOR_GROUP_SPLIT:
        return stratified_row_splits(
            df,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_state=random_state,
        )

    tr = float(train_ratio if train_ratio is not None else config.DATASET_TRAIN_RATIO)
    vr = float(val_ratio if val_ratio is not None else config.DATASET_VAL_RATIO)
    te = float(test_ratio if test_ratio is not None else config.DATASET_TEST_RATIO)
    rs = int(random_state if random_state is not None else config.RANDOM_STATE)

    src = df.groupby(group_col, observed=True).size().reset_index(name="n")
    src["label_raw"] = _source_level_labels(df, group_col).reindex(src[group_col].astype(str)).to_numpy()
    src[group_col] = src[group_col].astype(str)
    y_src = src["label_raw"].astype(str)

    tr_val_src, test_src = train_test_split(
        src[group_col],
        test_size=te,
        random_state=rs,
        stratify=y_src if y_src.nunique() > 1 else None,
    )
    tr_val_frame = src[src[group_col].isin(tr_val_src)]
    y_tr_val = tr_val_frame["label_raw"].astype(str)
    val_rel = vr / max(tr + vr, 1e-9)
    train_src, val_src = train_test_split(
        tr_val_src,
        test_size=val_rel,
        random_state=rs,
        stratify=y_tr_val if y_tr_val.nunique() > 1 else None,
    )

    train_df = df[df[group_col].astype(str).isin(train_src)].reset_index(drop=True)
    val_df = df[df[group_col].astype(str).isin(val_src)].reset_index(drop=True)
    test_df = df[df[group_col].astype(str).isin(test_src)].reset_index(drop=True)
    return train_df, val_df, test_df


def frames_to_bundle(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    note: str,
) -> DatasetBundle:
    cols = ["text_raw", "label_raw"]
    if "source_id" in train_df.columns:
        cols.append("source_id")
    return DatasetBundle(
        train=train_df[cols].copy(),
        test=test_df[cols].copy(),
        val=val_df[cols].copy(),
        text_column="text_raw",
        label_column="label_raw",
        source_note=note,
    )


def bundle_from_protocol(
    df: pd.DataFrame,
    protocol: SplitProtocol,
) -> DatasetBundle:
    if protocol == "stratified_row":
        tr, va, te = stratified_row_splits(df)
        note = "ablation split: stratified_row"
    else:
        tr, va, te = group_by_source_splits(df)
        note = "ablation split: group_by_source"
    return frames_to_bundle(tr, va, te, note=note)


def count_groups(df: pd.DataFrame, group_col: str = "source_id") -> int:
    if group_col not in df.columns:
        return 0
    return int(df[group_col].astype(str).nunique())
