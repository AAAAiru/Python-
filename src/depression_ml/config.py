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
MIN_TEXT_CHARS = 12  # hard minimum to run at all
MIN_TEXT_CHARS_SOFT = 40  # below: low confidence + tier guardrails
MIN_TEXT_CHARS_MEDIUM = 80  # below: medium confidence
MIN_WORDS_FOR_HIGH_CONF = 8
MIN_LATIN_LETTER_RATIO = 0.82  # see preprocess.looks_english
GUI_LEXICON_PREVIEW = 5  # max phrases per category in the GUI

# Post-model rules (short / positive snippets — reduce false medium/high)
# Allow strongly positive, lexicon-clean text in the lower part of the medium
# band to be corrected, while requiring clear rather than marginal positivity.
RISK_OVERRIDE_MAX_PROB = 0.45
RISK_OVERRIDE_MIN_SENTIMENT = 0.50
RISK_STRONG_POSITIVE_MIN_SENTIMENT = 0.75
RISK_STRONG_POSITIVE_MAX_PROB = 0.90
RISK_SHORT_MAX_PROB_FOR_HIGH = 0.88  # short text cannot be 高风险 below this prob without lexicon
RISK_HIGH_REQUIRES_LEXICON = True
RISK_HIGH_MIN_PROB_WITHOUT_LEXICON = 0.92

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

# --- Ablation study (scripts/run_ablation.py) ---
ABLATION_CV_FOLDS = 5

# --- Out-of-domain evaluation ---
OOV_TRAIN_SOURCES = ("reddit",)
OOV_TEST_SOURCES = ("depression_text_clf", "sentiment", "urdu")
AUTO_REBUILD_DATA = True

# --- Sentence embedding baseline (optional; requires sentence-transformers) ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# If set and folder contains config.json (from scripts/download_embedding_model.py), skip Hub download.
EMBEDDING_MODEL_LOCAL_DIR: Path | None = PROJECT_ROOT / "models" / "all-MiniLM-L6-v2"
# Used when HF_ENDPOINT env is unset (China mirror). See README §3.2.
HF_ENDPOINT_MIRROR = "https://hf-mirror.com"
HF_HUB_DOWNLOAD_TIMEOUT = 300
TRAIN_EMBEDDING_BASELINE = True
EMBEDDING_BATCH_SIZE = 64

# --- Ethics / documentation (written into artifacts/dataset_meta.json) ---
LABEL_DISCLAIMER = (
    "Labels reflect dataset authors / heuristics (e.g. Reddit self-report or Kaggle crowdsourcing), "
    "not clinical diagnosis. Do not use outputs for medical decisions or crisis triage."
)
