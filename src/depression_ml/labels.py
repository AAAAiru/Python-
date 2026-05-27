"""Shared label encoding helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def binary_labels(df: pd.DataFrame, positive_labels: set[str]) -> np.ndarray:
    col = df["label_raw"]
    num = pd.to_numeric(col, errors="coerce")
    if num.notna().all():
        u = np.unique(num.to_numpy(dtype=float))
        if u.size <= 2 and np.all(np.isin(u, (0.0, 1.0))):
            return (num.to_numpy(dtype=float) == 1.0).astype(int)
    return col.isin(positive_labels).astype(int).to_numpy()
