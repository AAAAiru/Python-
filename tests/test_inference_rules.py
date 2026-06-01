"""GUI inference guardrails for short / positive text."""

from __future__ import annotations


def test_positive_short_sentence_low_risk():
    from depression_ml.risk import assess_text
    from depression_ml.config import ARTIFACTS_DIR

    if not (ARTIFACTS_DIR / "depression_model.pkl").exists():
        return

    r = assess_text("i am happy and i love my cat", ARTIFACTS_DIR)
    assert r.tier == "低风险"
    assert r.prob < 0.35
    assert "positive_context_override" in r.flags or r.model_tier == "低风险"


def test_resolve_positive_benign_without_model():
    from depression_ml.risk import resolve_display_tier

    tier, model_tier, flags = resolve_display_tier(
        0.069,
        {"low": 0.35, "high": 0.70},
        clean="i am happy and i love my cat",
        sentiment=0.55,
        lex={"emnlp_mh_hits": 0, "emnlp_neg_diag": 0, "emnlp_pos_diag": 0},
        raw_len=30,
    )
    assert tier == "低风险"
    assert "positive_context_override" in flags


def test_short_text_caps_high_without_lexicon():
    from depression_ml.risk import resolve_display_tier

    tier, _, flags = resolve_display_tier(
        0.75,
        {"low": 0.35, "high": 0.70},
        clean="ok",
        sentiment=0.0,
        lex={"emnlp_mh_hits": 0, "emnlp_neg_diag": 0, "emnlp_pos_diag": 0},
        raw_len=2,
    )
    assert tier != "高风险"
    assert "short_text_cap_high" in flags or "short_or_sparse_text" in flags
