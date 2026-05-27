"""Project-wide defaults. Edit paths here after placing Kaggle CSVs under data/."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATASET_DB_FILENAME = "depression.db"

# Dataset cleaning when building SQLite / train|val|test CSV exports
MIN_DATASET_TEXT_CHARS = 20
DATASET_TRAIN_RATIO = 0.64
DATASET_VAL_RATIO = 0.16
DATASET_TEST_RATIO = 0.20

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

# --- EMNLP 2017 reference lexicon (local clone under reference/) ---
EMNLP17_USER_SELECTION_DIR = PROJECT_ROOT / "reference" / "emnlp17-depression" / "user_selection"
EMNLP17_UPSTREAM_URL = "https://github.com/Georgetown-IR-Lab/emnlp17-depression"
EMNLP17_LEXICON_FILES = (
    "mh_patterns.txt",
    "mh_subreddits.txt",
    "diagpatterns_positive.txt",
    "diagpatterns_negative.txt",
    "expansions.json",
)

# --- Out-of-domain evaluation ---
OOV_TRAIN_SOURCES = ("reddit",)
OOV_TEST_SOURCES = ("depression_text_clf", "sentiment", "urdu")
AUTO_REBUILD_DATA = True

# --- Sentence embedding baseline (optional; requires sentence-transformers) ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TRAIN_EMBEDDING_BASELINE = True
EMBEDDING_BATCH_SIZE = 64

# --- Ethics / documentation (written into artifacts/dataset_meta.json) ---
LABEL_DISCLAIMER = (
    "Labels reflect dataset authors / heuristics (e.g. Reddit self-report or Kaggle crowdsourcing), "
    "not clinical diagnosis. Do not use outputs for medical decisions or crisis triage."
)
