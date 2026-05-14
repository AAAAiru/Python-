"""Lexicon / diagnosis-style cues from EMNLP 2017 Reddit depression user-selection resources.

Reference (local copy): ``reference/emnlp17-depression/user_selection/``. These counts are
concatenated as dense numeric columns next to TF-IDF (see ``features.py``). They do not
replace the supervised model; they encode weak prior structure similar in spirit to the
original paper's user-selection stage.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Iterable

from . import config

_PLACEHOLDERS = ("_condition", "_doctor", "_ref")


def _lexicon_dir() -> Path:
    return config.PROJECT_ROOT / "reference" / "emnlp17-depression" / "user_selection"


def _read_nonempty_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return [ln.strip() for ln in lines if ln.strip()]


def _expand_line(line: str, expansions: dict[str, list[str]], *, max_combo: int) -> list[str]:
    """Expand underscore placeholders using Cartesian product (capped)."""
    s = line.lower().strip()
    if not s:
        return []
    present = [p for p in _PLACEHOLDERS if p in s]
    if not present:
        return [re.sub(r"\s+", " ", s)]
    pools: list[list[str]] = []
    for p in present:
        vals = expansions.get(p) or [p]
        pools.append([str(v).lower() for v in vals])
    out: list[str] = []
    for combo in product(*pools):
        if len(out) >= max_combo:
            break
        t = s
        for ph, val in zip(present, combo):
            t = t.replace(ph, val)
        t = re.sub(r"\s+", " ", t.strip())
        if len(t) >= 4:
            out.append(t)
    return out


def _build_expanded_phrases(lines: Iterable[str], expansions: dict[str, list[str]], *, max_combo: int) -> frozenset[str]:
    bag: set[str] = set()
    for raw in lines:
        for phrase in _expand_line(raw, expansions, max_combo=max_combo):
            bag.add(phrase)
    return frozenset(bag)


@lru_cache(maxsize=1)
def _compiled_resources() -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    root = _lexicon_dir()
    if not root.exists():
        return frozenset(), frozenset(), frozenset()

    exp_path = root / "expansions.json"
    expansions: dict[str, list[str]] = {}
    if exp_path.exists():
        expansions = json.loads(exp_path.read_text(encoding="utf-8"))

    mh_terms = frozenset(t for t in _read_nonempty_lines(root / "mh_patterns.txt") if len(t) >= 2)

    pos_lines = _read_nonempty_lines(root / "diagpatterns_positive.txt")
    neg_lines = _read_nonempty_lines(root / "diagpatterns_negative.txt")

    pos_phrases = _build_expanded_phrases(pos_lines, expansions, max_combo=80)
    neg_phrases = _build_expanded_phrases(neg_lines, expansions, max_combo=80)
    return mh_terms, pos_phrases, neg_phrases


def extract_emnlp17_detailed(
    text: str,
    *,
    collect_matches: bool = False,
    match_limit: int = 8,
) -> dict[str, float | list[str]]:
    """Counts plus optional capped lists of matched phrases / terms."""
    lim = max(0, int(match_limit))
    empty: dict[str, float | list[str]] = {
        "emnlp_mh_hits": 0.0,
        "emnlp_pos_diag": 0.0,
        "emnlp_neg_diag": 0.0,
        "mh_matches": [],
        "pos_matches": [],
        "neg_matches": [],
    }
    if not text:
        return empty
    mh_terms, pos_phrases, neg_phrases = _compiled_resources()
    if not mh_terms and not pos_phrases and not neg_phrases:
        return empty

    tpad = f" {text} "
    mh_hits = 0
    pos_hits = 0
    neg_hits = 0
    mh_m: list[str] = []
    pos_m: list[str] = []
    neg_m: list[str] = []

    for term in mh_terms:
        if len(term) < 2:
            continue
        if f" {term.lower()} " in tpad:
            mh_hits += 1
            if collect_matches and len(mh_m) < lim:
                mh_m.append(term)

    for phrase in pos_phrases:
        if phrase and phrase in text:
            pos_hits += 1
            if collect_matches and len(pos_m) < lim:
                pos_m.append(phrase[:120])

    for phrase in neg_phrases:
        if phrase and phrase in text:
            neg_hits += 1
            if collect_matches and len(neg_m) < lim:
                neg_m.append(phrase[:120])

    return {
        "emnlp_mh_hits": float(mh_hits),
        "emnlp_pos_diag": float(pos_hits),
        "emnlp_neg_diag": float(neg_hits),
        "mh_matches": mh_m,
        "pos_matches": pos_m,
        "neg_matches": neg_m,
    }


def extract_emnlp17_features(text: str) -> dict[str, float]:
    """Return fixed keys used by ``features.STAT_FEATURE_ORDER``."""
    d = extract_emnlp17_detailed(text, collect_matches=False)
    return {
        "emnlp_mh_hits": float(d["emnlp_mh_hits"]),
        "emnlp_pos_diag": float(d["emnlp_pos_diag"]),
        "emnlp_neg_diag": float(d["emnlp_neg_diag"]),
    }
