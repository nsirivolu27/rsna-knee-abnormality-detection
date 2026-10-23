"""Load and summarize the verified exam-level label columns."""

from pathlib import Path
from typing import NamedTuple, Optional, Union

import pandas as pd

from . import config


class Labels(NamedTuple):
    """Two aligned frames containing label values and observation status."""

    values: pd.DataFrame
    observed: pd.DataFrame


CsvPath = Union[Path, str]


def load_labels(csv_path: Optional[CsvPath] = None) -> Labels:
    """Load train labels and return values together with an observed mask.

    Label values remain 0.0, 1.0, or NaN. NaN means the label was not
    observed; it is never converted to a negative label. StudyInstanceUID is
    used as the unique index and duplicate exam IDs fail loudly.
    """
    path = Path(csv_path) if csv_path is not None else config.TRAIN_CSV
    columns = [config.EXAM_ID_COLUMN, *config.TARGET_LABELS]
    frame = pd.read_csv(path, usecols=columns)
    frame = frame.set_index(config.EXAM_ID_COLUMN, verify_integrity=True)

    values = frame.loc[:, config.TARGET_LABELS].astype(float)
    invalid_values = values.notna() & ~values.isin([0.0, 1.0])
    if invalid_values.any().any():
        for column in values.columns:
            invalid = values.loc[invalid_values[column], column]
            if not invalid.empty:
                bad_values = sorted(invalid.unique().tolist(), key=str)
                raise ValueError(
                    f"Invalid value(s) in label column {column!r}: "
                    f"{bad_values!r}. Expected 0.0, 1.0, or NaN."
                )

    observed = values.notna()
    labels = Labels(values=values, observed=observed)

    if not labels.values.index.equals(labels.observed.index):
        raise RuntimeError("Label values and observed mask indexes are misaligned.")
    if not labels.values.columns.equals(labels.observed.columns):
        raise RuntimeError("Label values and observed mask columns are misaligned.")

    return labels


def summarize_labels(labels: Labels) -> pd.DataFrame:
    """Return positive, negative, and missing counts for every label."""
    values = labels.values
    observed = labels.observed

    positive = ((values == 1.0) & observed).sum(axis=0)
    negative = ((values == 0.0) & observed).sum(axis=0)
    missing = (~observed).sum(axis=0)

    return pd.DataFrame(
        {
            "positive": positive.astype(int),
            "negative": negative.astype(int),
            "missing": missing.astype(int),
        }
    )


def fully_labeled_exams(labels: Labels) -> pd.DataFrame:
    """Return exams indexed by StudyInstanceUID for extractor validation.

    These exams are meant to be joined against reports for extractor
    validation. This is an extractor-validation set, not a training set.
    """
    complete_rows = labels.observed.all(axis=1)
    return labels.values.loc[complete_rows].copy()
