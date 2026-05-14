"""Project-wide defaults. Edit paths here after placing Kaggle CSVs under data/."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# TF-IDF (keep moderate dimensionality for laptop RAM with dense tree models)
TFIDF_MAX_FEATURES = 6000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 3
TFIDF_MAX_DF = 0.9

# Stratified subsample of the training split for model fitting (full data still used for val/test)
MAX_TRAIN_ROWS = 24000

# Train/val when no dedicated val set
VAL_FRACTION = 0.1
RANDOM_STATE = 42

# Binary task: which raw labels count as positive (depression tendency)
POSITIVE_LABELS = {"Depression"}
# Set to {"Depression", "Suicidal"} if your report wants broader "clinical distress"
# POSITIVE_LABELS = {"Depression", "Suicidal"}

# Risk buckets: tuned on validation set by train.py (saved to artifacts)
RISK_LOW_DEFAULT = 0.35
RISK_HIGH_DEFAULT = 0.70

# Reddit-only: keep only Depression vs Normal rows
REDDIT_BINARY_FILTER = True

# --- Inference UX (GUI / CLI) ---
MIN_TEXT_CHARS = 12
MIN_LATIN_LETTER_RATIO = 0.82  # see preprocess.looks_english
GUI_LEXICON_PREVIEW = 5  # max phrases per category in the GUI

# --- Post-training probability calibration (validation Platt) ---
USE_PLATT_CALIBRATION = True

# --- Ethics / documentation (written into artifacts/dataset_meta.json) ---
LABEL_DISCLAIMER = (
    "Labels reflect dataset authors / heuristics (e.g. Reddit self-report or Kaggle crowdsourcing), "
    "not clinical diagnosis. Do not use outputs for medical decisions or crisis triage."
)
