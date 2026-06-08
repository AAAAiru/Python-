"""Dataset quality helpers: stable fingerprints and lightweight near-duplicate checks."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Iterable

import pandas as pd


_SPACE_RE = re.compile(r"\s+")


def normalized_text(text: object) -> str:
    return _SPACE_RE.sub(" ", str(text).strip().lower())


def text_fingerprint(text: object) -> str:
    return hashlib.sha256(normalized_text(text).encode("utf-8")).hexdigest()


def token_shingles(text: object, size: int = 3) -> set[str]:
    words = normalized_text(text).split()
    if len(words) < size:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + size]) for i in range(len(words) - size + 1)}


def near_duplicate_count(texts: Iterable[object], threshold: float = 0.9) -> int:
    """Estimate near duplicates using shared 3-gram buckets, avoiding all-pairs comparison."""
    buckets: dict[str, list[tuple[int, set[str]]]] = {}
    saved_by_index: dict[int, set[str]] = {}
    duplicates = 0
    for idx, text in enumerate(texts):
        shingles = token_shingles(text)
        if not shingles:
            continue
        candidates: set[int] = set()
        for shingle in shingles:
            candidates.update(item[0] for item in buckets.get(shingle, ()))
        matched = False
        for candidate in candidates:
            other = saved_by_index[candidate]
            union = len(shingles | other)
            if union and len(shingles & other) / union >= threshold:
                matched = True
                break
        if matched:
            duplicates += 1
        saved_by_index[idx] = shingles
        for shingle in shingles:
            buckets.setdefault(shingle, []).append((idx, shingles))
    return duplicates


def dataset_quality_summary(df: pd.DataFrame) -> dict:
    fingerprints = df["text_clean"].map(text_fingerprint)
    exact_duplicates = int(fingerprints.duplicated().sum())
    lengths = df["text_clean"].astype(str).str.len()
    by_source = {}
    if "source_id" in df.columns:
        for source, part in df.groupby("source_id", dropna=False):
            by_source[str(source)] = {
                "rows": int(len(part)),
                "label_counts": {
                    str(k): int(v) for k, v in Counter(part["label_raw"].astype(str)).items()
                },
                "mean_chars": float(part["text_clean"].astype(str).str.len().mean()),
            }
    return {
        "rows": int(len(df)),
        "exact_duplicate_fingerprints": exact_duplicates,
        "near_duplicate_estimate": int(near_duplicate_count(df["text_clean"])),
        "char_length": {
            "min": int(lengths.min()) if len(lengths) else 0,
            "median": float(lengths.median()) if len(lengths) else 0.0,
            "mean": float(lengths.mean()) if len(lengths) else 0.0,
            "max": int(lengths.max()) if len(lengths) else 0,
        },
        "per_source": by_source,
    }
