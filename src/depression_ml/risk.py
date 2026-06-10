"""Map model scores to low / medium / high risk tiers with GUI-safe inference rules."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from . import config
from .features import compound_sentiment, vectorize_single
from .preprocess import preprocess_text_en
from .probability_calibrate import apply_platt, load_platt_optional

# Lightweight English positive cues (not in training labels; inference-only guardrail).
_POSITIVE_CUE_RE = re.compile(
    r"\b(happy|happiness|joy|love|loved|loving|enjoy|enjoyed|grateful|gratitude|"
    r"wonderful|great|amazing|excited|glad|pleased|fun|smile|smiling|blessed|"
    r"content|cheerful|delighted|fantastic|awesome|good day|feel good)\b",
    re.IGNORECASE,
)

# Explicit first-person crisis language needs a safety response even when the
# older EMNLP lexicon does not contain the wording.
_CRISIS_CUE_RE = re.compile(
    r"\b(?:"
    r"(?:kill|hurt|harm)\s+myself|"
    r"end\s+my\s+(?:life|existence)|"
    r"take\s+my\s+own\s+life|"
    r"(?:i\s+)?(?:want|wish|plan|intend|think|thinking|thought|thoughts)\s+"
    r"(?:about\s+|of\s+|to\s+)?(?:die|dying|suicide|killing\s+myself|hurting\s+myself|harming\s+myself)|"
    r"i\s+(?:do\s+not|don['’]?t)\s+want\s+to\s+(?:live|be\s+alive|continue)|"
    r"(?:i\s+)?(?:have|feel\s+there\s+is)\s+no\s+reason\s+to\s+(?:live|continue)|"
    r"i(?:'m|\s+am)\s+(?:better\s+off\s+dead|suicidal)|"
    r"i\s+(?:want|wish)\s+i\s+(?:would\s+not|wouldn['’]?t|did\s+not|didn['’]?t)\s+wake\s+up|"
    r"self[\s-]?harm|suicidal"
    r")\b",
    re.IGNORECASE,
)


@dataclass
class AssessmentResult:
    """Full GUI/CLI assessment with explainability flags."""

    tier: str
    prob: float
    model_tier: str
    thresholds: dict[str, float]
    confidence: str  # 低 | 中 | 高
    flags: list[str] = field(default_factory=list)
    sentiment_compound: float = 0.0
    char_len: int = 0
    word_count: int = 0
    lex: dict[str, Any] = field(default_factory=dict)

    @property
    def flag_notes_zh(self) -> str:
        notes = {
            "positive_context_override": "文本偏积极且缺乏心理困扰词，已下调风险档位",
            "short_text_cap_high": "文本较短，未检出明显困扰词，已限制最高档位",
            "short_positive_downgrade": "短文本且语境积极，模型分数参考性有限",
            "high_tier_requires_lexicon": "未检出明显心理/负向词典命中，已限制为高中档",
            "short_or_sparse_text": "文本过短或词数过少，结果仅供参考",
            "explicit_crisis_language": "检测到明确的自伤或自杀相关表达，已优先显示安全求助提示",
        }
        parts = [notes.get(f, f) for f in self.flags]
        return "；".join(parts) if parts else ""


def load_risk_thresholds(artifacts_dir: Path) -> dict[str, float]:
    path = artifacts_dir / "risk_thresholds.json"
    if not path.exists():
        return normalize_risk_thresholds(
            {"low": config.RISK_LOW_DEFAULT, "high": config.RISK_HIGH_DEFAULT, "operating_threshold": 0.5}
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    return normalize_risk_thresholds(raw)


def normalize_risk_thresholds(thresholds: dict[str, float]) -> dict[str, float]:
    """Fix pathological tertiles (e.g. low≈0.04, high≈0.97) that mark almost everything as medium."""
    low = float(thresholds.get("low", config.RISK_LOW_DEFAULT))
    high = float(thresholds.get("high", config.RISK_HIGH_DEFAULT))
    if low < 0.15 and high > 0.90:
        low = config.RISK_LOW_DEFAULT
        high = config.RISK_HIGH_DEFAULT
    if low >= high:
        low, high = config.RISK_LOW_DEFAULT, config.RISK_HIGH_DEFAULT
    out = dict(thresholds)
    out["low"] = low
    out["high"] = high
    return out


def depression_probability(
    text: str,
    model: Any,
    vectorizer: Any,
    scaler: Any,
    platt: Any | None = None,
) -> float:
    clean = preprocess_text_en(text)
    if vectorizer is None or scaler is None:
        p = float(model.predict_proba([clean])[0, 1])
    else:
        Xs = vectorize_single(vectorizer, scaler, clean)
        if hasattr(model, "predict_proba"):
            p = float(model.predict_proba(Xs)[0, 1])
        else:
            scores = model.decision_function(Xs)
            s = float(scores[0])
            p = float(1.0 / (1.0 + np.exp(-s)))
    if platt is not None:
        p = apply_platt(platt, p)
    return p


def risk_tier(prob: float, thresholds: dict[str, float]) -> str:
    """Model-only tier from probability cutoffs (no GUI guardrails)."""
    thr = normalize_risk_thresholds(thresholds)
    low, high = float(thr["low"]), float(thr["high"])
    if prob < low:
        return "低风险"
    if prob > high:
        return "高风险"
    return "中风险"


def _has_positive_cue_words(clean: str) -> bool:
    return bool(_POSITIVE_CUE_RE.search(clean))


def _has_explicit_crisis_language(clean: str) -> bool:
    return bool(_CRISIS_CUE_RE.search(clean))


def _lex_hits(lex: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(lex.get("emnlp_mh_hits", 0)),
        float(lex.get("emnlp_neg_diag", 0)),
        float(lex.get("emnlp_pos_diag", 0)),
    )


def _is_clearly_positive_benign(
    prob: float,
    sentiment: float,
    lex: dict[str, Any],
    clean: str,
) -> bool:
    mh, neg, pos = _lex_hits(lex)
    if mh >= 1 or neg >= 1:
        return False
    explicit_positive = pos >= 1 or _has_positive_cue_words(clean)
    strong_positive = (
        explicit_positive
        and sentiment >= config.RISK_STRONG_POSITIVE_MIN_SENTIMENT
        and prob < config.RISK_STRONG_POSITIVE_MAX_PROB
    )
    if prob >= config.RISK_OVERRIDE_MAX_PROB and not strong_positive:
        return False
    positive_lang = (
        sentiment >= config.RISK_OVERRIDE_MIN_SENTIMENT
        or explicit_positive
    )
    return positive_lang


def _infer_confidence(raw_len: int, word_count: int) -> str:
    if raw_len < config.MIN_TEXT_CHARS_SOFT or word_count < config.MIN_WORDS_FOR_HIGH_CONF:
        return "低"
    if raw_len < config.MIN_TEXT_CHARS_MEDIUM:
        return "中"
    return "高"


def resolve_display_tier(
    prob: float,
    thresholds: dict[str, float],
    *,
    clean: str,
    sentiment: float,
    lex: dict[str, Any],
    raw_len: int,
) -> tuple[str, str, list[str]]:
    """
    Apply post-model rules for GUI display.

    Returns (display_tier, model_tier, flags).
    """
    thr = normalize_risk_thresholds(thresholds)
    flags: list[str] = []
    model_tier = risk_tier(prob, thr)
    tier = model_tier

    word_count = len(clean.split())
    mh, neg, pos = _lex_hits(lex)
    short_or_sparse = raw_len < config.MIN_TEXT_CHARS_SOFT or word_count < config.MIN_WORDS_FOR_HIGH_CONF
    if short_or_sparse:
        flags.append("short_or_sparse_text")

    if _has_explicit_crisis_language(clean):
        flags.append("explicit_crisis_language")
        return "高风险", model_tier, flags

    if _is_clearly_positive_benign(prob, sentiment, lex, clean):
        if tier != "低风险":
            flags.append("positive_context_override")
        tier = "低风险"

    if short_or_sparse:
        if tier == "高风险" and prob < config.RISK_SHORT_MAX_PROB_FOR_HIGH and mh < 1 and neg < 1:
            tier = "中风险"
            flags.append("short_text_cap_high")
        if tier == "中风险" and _is_clearly_positive_benign(prob, sentiment, lex, clean):
            tier = "低风险"
            flags.append("short_positive_downgrade")

    if config.RISK_HIGH_REQUIRES_LEXICON and tier == "高风险":
        if mh < 1 and neg < 1 and prob < config.RISK_HIGH_MIN_PROB_WITHOUT_LEXICON:
            tier = "中风险"
            flags.append("high_tier_requires_lexicon")

    return tier, model_tier, flags


def assess_text(
    text: str,
    artifacts_dir: Path,
    *,
    model: Any | None = None,
    vectorizer: Any | None = None,
    scaler: Any | None = None,
    thresholds: dict[str, float] | None = None,
    platt: Any | None = None,
    collect_lexicon_matches: bool = True,
) -> AssessmentResult:
    """End-to-end assessment with lexicon features and GUI guardrails."""
    from .emnlp17_signals import extract_emnlp17_detailed

    if model is None or vectorizer is None or scaler is None or thresholds is None:
        model, vectorizer, scaler, thresholds, platt = load_artifacts(artifacts_dir)
    elif platt is None:
        platt = load_platt_optional(artifacts_dir)

    clean = preprocess_text_en(text)
    raw_len = len(text.strip())
    word_count = len(clean.split())

    det = extract_emnlp17_detailed(
        clean,
        collect_matches=collect_lexicon_matches,
        match_limit=config.GUI_LEXICON_PREVIEW,
    )
    sentiment = compound_sentiment(clean)
    prob = depression_probability(text, model, vectorizer, scaler, platt)

    tier, model_tier, flags = resolve_display_tier(
        prob,
        thresholds,
        clean=clean,
        sentiment=sentiment,
        lex=det,
        raw_len=raw_len,
    )
    confidence = _infer_confidence(raw_len, word_count)

    return AssessmentResult(
        tier=tier,
        prob=prob,
        model_tier=model_tier,
        thresholds=normalize_risk_thresholds(thresholds),
        confidence=confidence,
        flags=flags,
        sentiment_compound=sentiment,
        char_len=raw_len,
        word_count=word_count,
        lex=det,
    )


def load_artifacts(artifacts_dir: Path):
    pipeline_path = artifacts_dir / "model_pipeline.pkl"
    if pipeline_path.exists():
        model = joblib.load(pipeline_path)
        vectorizer = scaler = None
    else:
        model = joblib.load(artifacts_dir / "depression_model.pkl")
        vectorizer = joblib.load(artifacts_dir / "tfidf.pkl")
        scaler = joblib.load(artifacts_dir / "scaler.pkl")
    thresholds = load_risk_thresholds(artifacts_dir)
    platt = load_platt_optional(artifacts_dir)
    return model, vectorizer, scaler, thresholds, platt


def assess(text: str, artifacts_dir: Path) -> tuple[str, float, dict[str, float]]:
    r = assess_text(text, artifacts_dir, collect_lexicon_matches=False)
    return r.tier, r.prob, r.thresholds


def assess_with_lexicon(text: str, artifacts_dir: Path) -> tuple[str, float, dict[str, float], dict[str, float]]:
    r = assess_text(text, artifacts_dir)
    lex = {
        k: float(r.lex[k])
        for k in (
            "emnlp_mh_hits",
            "emnlp_pos_diag",
            "emnlp_neg_diag",
            "emnlp_subreddit_word_hits",
            "emnlp_subreddit_r_hits",
        )
    }
    return r.tier, r.prob, r.thresholds, lex
