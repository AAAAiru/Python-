"""Load and normalize multiple mental-health CSV sources into a unified frame."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from . import config
from .io_data import _normalize_columns, _pick_column, _read_csv, TEXT_CANDIDATES, LABEL_CANDIDATES
from .preprocess import looks_english, preprocess_text_en


@dataclass(frozen=True)
class SourceSpec:
    id: str
    file: str
    text_column: str | None = None
    text_columns: tuple[str, ...] = ()
    label_column: str = "label"
    label_type: str = "binary_numeric"
    alt_files: tuple[str, ...] = ()
    required: bool = False
    min_chars: int | None = None


def _manifest_path(data_dir: Path) -> Path:
    return data_dir / "sources.json"


def load_manifest(data_dir: Path | None = None) -> list[SourceSpec]:
    data_dir = data_dir or config.DATA_DIR
    path = _manifest_path(data_dir)
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    specs: list[SourceSpec] = []
    for item in raw.get("sources", []):
        text_cols = item.get("text_columns") or []
        specs.append(
            SourceSpec(
                id=str(item["id"]),
                file=str(item["file"]),
                text_column=item.get("text_column"),
                text_columns=tuple(text_cols),
                label_column=str(item.get("label_column", "label")),
                label_type=str(item.get("label_type", "binary_numeric")),
                alt_files=tuple(item.get("alt_files") or []),
                required=bool(item.get("required", False)),
                min_chars=item.get("min_chars"),
            )
        )
    return specs


def resolve_source_path(data_dir: Path, spec: SourceSpec) -> Path | None:
    for name in (spec.file, *spec.alt_files):
        p = data_dir / name
        if p.exists():
            return p
    return None


def _binary_from_status(status: object, positive_labels: set[str]) -> str:
    s = str(status).strip()
    return "1" if s in positive_labels else "0"


def _binary_from_urdu_severity(label: object) -> str:
    try:
        v = int(float(label))
    except (TypeError, ValueError):
        return "0"
    return "1" if v >= 1 else "0"


def _pick_text_series(df: pd.DataFrame, spec: SourceSpec) -> pd.Series:
    if spec.text_column and spec.text_column in df.columns:
        return df[spec.text_column].astype("string")
    if spec.text_columns:
        series: pd.Series | None = None
        for col in spec.text_columns:
            if col not in df.columns:
                continue
            s = df[col].astype("string")
            series = s if series is None else series.fillna(s)
        if series is not None:
            return series
    t = _pick_column(df, TEXT_CANDIDATES)
    if t:
        return df[t].astype("string")
    raise ValueError(f"Cannot find text column for source {spec.id}: {list(df.columns)}")


def _pick_label_series(df: pd.DataFrame, spec: SourceSpec, positive_labels: set[str]) -> pd.Series:
    col = spec.label_column
    if col not in df.columns:
        l = _pick_column(df, LABEL_CANDIDATES)
        if not l:
            raise ValueError(f"Cannot find label column for source {spec.id}")
        col = l
    raw = df[col]

    if spec.label_type == "binary_numeric":
        num = pd.to_numeric(raw, errors="coerce")
        return (num.fillna(0).astype(int) == 1).astype(int).astype(str)
    if spec.label_type == "mental_health_status":
        return raw.map(lambda x: _binary_from_status(x, positive_labels))
    if spec.label_type == "urdu_severity":
        return raw.map(_binary_from_urdu_severity)
    if spec.label_type == "string_positive_set":
        return raw.map(lambda x: _binary_from_status(x, positive_labels))
    raise ValueError(f"Unknown label_type: {spec.label_type}")


def load_source_frame(
    path: Path,
    spec: SourceSpec,
    *,
    positive_labels: set[str] | None = None,
    english_only: bool = True,
    min_chars: int = 0,
) -> pd.DataFrame:
    """Return columns: text_raw, label_raw, source_id, source_file."""
    positive_labels = positive_labels or config.POSITIVE_LABELS
    df = _read_csv(path)
    text = _pick_text_series(df, spec)
    labels = _pick_label_series(df, spec, positive_labels)

    out = pd.DataFrame(
        {
            "text_raw": text,
            "label_raw": labels.astype(str),
            "source_id": spec.id,
            "source_file": path.name,
        }
    )
    out = out[out["text_raw"].notna() & (out["text_raw"].str.len() > 0)]
    out["text_clean"] = out["text_raw"].map(preprocess_text_en)
    out["char_len"] = out["text_clean"].str.len()
    out = out[out["char_len"] >= int(min_chars)]

    if english_only:
        mask = out["text_raw"].map(lambda t: looks_english(str(t)))
        out = out[mask]

    out = out.drop_duplicates(subset=["text_clean"], keep="first")
    return out.reset_index(drop=True)


def merge_source_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    parts = [f for f in frames if f is not None and len(f) > 0]
    if not parts:
        raise ValueError("No source frames to merge.")
    combined = pd.concat(parts, ignore_index=True)
    combined = combined.drop_duplicates(subset=["text_clean"], keep="first")
    return combined.reset_index(drop=True)


def discover_and_load(
    data_dir: Path | None = None,
    *,
    source_ids: list[str] | None = None,
    english_only: bool = True,
    min_chars: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load all available sources from manifest; skip missing optional files."""
    data_dir = data_dir or config.DATA_DIR
    default_min = int(min_chars if min_chars is not None else config.MIN_DATASET_TEXT_CHARS)
    manifest = load_manifest(data_dir)
    if not manifest:
        raise FileNotFoundError(f"No sources.json in {data_dir}")

    want = set(source_ids) if source_ids else {s.id for s in manifest}
    frames: list[pd.DataFrame] = []
    report: dict[str, Any] = {"loaded": [], "skipped": [], "missing_required": []}

    for spec in manifest:
        if spec.id not in want:
            continue
        path = resolve_source_path(data_dir, spec)
        if path is None:
            entry = {"id": spec.id, "expected": spec.file, "alt_files": list(spec.alt_files)}
            if spec.required:
                report["missing_required"].append(entry)
            else:
                report["skipped"].append({**entry, "reason": "file_not_found"})
            continue
        effective_min = int(spec.min_chars if spec.min_chars is not None else default_min)
        frame = load_source_frame(
            path, spec, english_only=english_only, min_chars=effective_min
        )
        frames.append(frame)
        report["loaded"].append(
            {
                "id": spec.id,
                "file": path.name,
                "rows": len(frame),
                "positives": int((frame["label_raw"] == "1").sum()),
            }
        )

    if report["missing_required"]:
        missing = ", ".join(m["id"] for m in report["missing_required"])
        raise FileNotFoundError(
            f"Required source file(s) missing: {missing}. See data/sources.json and data/README.md."
        )
    if not frames:
        raise FileNotFoundError(
            "No optional source files found. Download CSVs listed in data/sources.json "
            "or pass --source for a single-file build."
        )

    merged = merge_source_frames(frames)
    report["rows_merged"] = len(merged)
    report["label_counts"] = merged["label_raw"].value_counts().astype(int).to_dict()
    return merged, report
