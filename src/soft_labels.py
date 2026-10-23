"""Load externally generated probabilistic labels without generating them."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import config


CONFIDENCE_SUFFIX = "_confidence"
RATIONALE_COLUMNS = ("rationale", "Rationale")


def _read_label_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Soft-label file does not exist: {path}")
    if path.suffix.casefold() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _with_exam_column(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if config.EXAM_ID_COLUMN not in result.columns:
        if result.index.name == config.EXAM_ID_COLUMN:
            result = result.reset_index()
        else:
            raise ValueError(f"Soft labels must contain {config.EXAM_ID_COLUMN!r}.")
    if result[config.EXAM_ID_COLUMN].isna().any():
        position = int(np.flatnonzero(result[config.EXAM_ID_COLUMN].isna().to_numpy())[0])
        raise ValueError(f"Missing exam ID in soft-label row {position}.")
    result[config.EXAM_ID_COLUMN] = result[config.EXAM_ID_COLUMN].astype(str)
    duplicate = result[config.EXAM_ID_COLUMN].duplicated(keep=False)
    if duplicate.any():
        duplicate_ids = result.loc[duplicate, config.EXAM_ID_COLUMN].drop_duplicates().tolist()
        raise ValueError(f"Duplicate soft-label exam ID(s): {duplicate_ids!r}")
    return result


def _train_exam_ids(train_csv: Optional[Path | str]) -> set[str]:
    path = Path(train_csv) if train_csv is not None else config.TRAIN_CSV
    train = pd.read_csv(path, usecols=[config.EXAM_ID_COLUMN])
    return set(train[config.EXAM_ID_COLUMN].dropna().astype(str))


def _validate_probability_columns(frame: pd.DataFrame, columns: tuple[str, ...], kind: str) -> None:
    raw = frame.loc[:, list(columns)]
    values = raw.apply(pd.to_numeric, errors="coerce")
    for column in columns:
        numeric = values[column]
        array = numeric.to_numpy(dtype=float)
        invalid = raw[column].notna().to_numpy() & (
            numeric.isna().to_numpy()
            | ~np.isfinite(array)
            | (array < 0.0)
            | (array > 1.0)
        )
        if invalid.any():
            position = int(np.flatnonzero(invalid)[0])
            exam_id = frame.iloc[position][config.EXAM_ID_COLUMN]
            raise ValueError(
                f"Invalid {kind} at row {position}, exam {exam_id!r}, column {column!r}: "
                f"{raw.iloc[position][column]!r}. Expected a value in [0, 1]."
            )


def validate_soft_labels(
    frame: pd.DataFrame,
    train_csv: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Validate and return a normalized, exam-indexed soft-label frame.

    Every row must identify an exam present in train.csv, contain all twelve
    probability columns, and contain no duplicate exam IDs. Optional
    per-label confidence columns use the label name plus _confidence or
    confidence_ plus the label name. Optional rationale is preserved.
    """
    result = _with_exam_column(frame)
    missing = [label for label in config.TARGET_LABELS if label not in result.columns]
    if missing:
        raise ValueError(f"Soft labels are missing required columns: {missing!r}")

    known_ids = _train_exam_ids(train_csv)
    unknown = sorted(set(result[config.EXAM_ID_COLUMN]) - known_ids)
    if unknown:
        first = unknown[0]
        row = int(result.index[result[config.EXAM_ID_COLUMN] == first][0])
        raise ValueError(
            f"Soft-label row {row} names exam {first!r}, which is absent from train.csv."
        )

    _validate_probability_columns(result, config.TARGET_LABELS, "probability")
    confidence_columns = [
        column
        for label in config.TARGET_LABELS
        for column in (f"{label}{CONFIDENCE_SUFFIX}", f"confidence_{label}")
        if column in result.columns
    ]
    _validate_probability_columns(result, tuple(confidence_columns), "confidence")

    normalized = result.set_index(config.EXAM_ID_COLUMN, verify_integrity=True)
    normalized.loc[:, config.TARGET_LABELS] = normalized.loc[:, config.TARGET_LABELS].apply(
        pd.to_numeric, errors="raise"
    ).astype(float)
    return normalized


def load_soft_labels(
    path: Path | str,
    train_csv: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Load and validate a CSV or parquet file of external soft labels."""
    return validate_soft_labels(_read_label_file(Path(path)), train_csv=train_csv)
