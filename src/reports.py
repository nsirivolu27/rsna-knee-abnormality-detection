"""Load reports and perform deliberately narrow text normalization."""

import re
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from . import config


CsvPath = Union[Path, str]

PLACEHOLDER_TOKENS: tuple[str, ...] = (
    "[DATE]",
    "[TIME]",
    "[REDACTED]",
    "[ID]",
    "[NAME]",
    "[YEAR]",
    "[IDENTIFIER]",
    "[PROFESSION]",
    "[AFFILIATION]",
)

_PLACEHOLDER_PATTERN = re.compile(
    "|".join(re.escape(token) for token in PLACEHOLDER_TOKENS)
)


def repair_deidentification(text: object) -> str:
    """Repair the systematic de-identification corruption before other edits."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        raw_text = ""
    else:
        raw_text = str(text)
    return re.sub(r"intact(?=\d)", "0.", raw_text)


def placeholder_signature(text: object) -> str:
    """Return the sorted placeholder signature present in one report."""
    repaired_text = repair_deidentification(text)
    present = [token for token in PLACEHOLDER_TOKENS if token in repaired_text]
    return "+".join(present) if present else "none"


def normalize_report(text: object, mask_placeholders: bool = False) -> str:
    """Repair corruption, optionally mask placeholders, and normalize whitespace."""
    repaired_text = repair_deidentification(text)
    if mask_placeholders:
        repaired_text = _PLACEHOLDER_PATTERN.sub("[MASKED]", repaired_text)
    return re.sub(r"\s+", " ", repaired_text).strip()


def detect_language(text: object) -> Optional[str]:
    """Detect a report language when langdetect is installed, otherwise return None."""
    try:
        from langdetect import detect
    except ImportError:
        return None

    try:
        return detect(normalize_report(text)[:400])
    except Exception:
        return None


def load_reports(
    csv_path: Optional[CsvPath] = None,
    mask_placeholders: bool = False,
    detect_languages: bool = False,
) -> pd.DataFrame:
    """Load reports with constrained normalization and optional language detection."""
    path = Path(csv_path) if csv_path is not None else config.TRAIN_CSV
    frame = pd.read_csv(path, usecols=[config.EXAM_ID_COLUMN, config.REPORT_COLUMN])
    frame = frame.set_index(config.EXAM_ID_COLUMN, verify_integrity=True)

    frame["PlaceholderSignature"] = frame[config.REPORT_COLUMN].map(
        placeholder_signature
    )
    frame[config.REPORT_COLUMN] = frame[config.REPORT_COLUMN].map(
        lambda text: normalize_report(text, mask_placeholders=mask_placeholders)
    )

    if detect_languages:
        frame["Language"] = frame[config.REPORT_COLUMN].map(detect_language)

    return frame
