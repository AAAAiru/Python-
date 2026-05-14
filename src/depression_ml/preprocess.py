"""English text preprocessing for social-media style posts."""

from __future__ import annotations

import re

import pandas as pd


_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_NON_ALPHA_RE = re.compile(r"[^a-z\s]+")


def preprocess_text_en(text: object) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text).lower()
    s = _URL_RE.sub(" ", s)
    s = _NON_ALPHA_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def looks_english(text: str, sample_chars: int = 200, min_ratio: float | None = None) -> bool:
    """Heuristic: Latin letters dominate the sample (robust to punctuation)."""
    if min_ratio is None:
        from . import config

        min_ratio = float(getattr(config, "MIN_LATIN_LETTER_RATIO", 0.85))
    if not text:
        return True
    chunk = text[:sample_chars]
    letters = sum(1 for c in chunk if c.isalpha())
    latin = sum(1 for c in chunk if "a" <= c.lower() <= "z")
    if letters == 0:
        return True
    return (latin / letters) >= min_ratio
