"""Construction and validation of the competition submission file.

The submission contract is narrow and unforgiving: exact column names in exact
order, one row per test examination, and every value a finite probability. This
module owns that contract so no other part of the pipeline has to know it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from . import config

SUBMISSION_COLUMNS: tuple[str, ...] = (config.EXAM_ID_COLUMN,) + config.TARGET_LABELS


def build_submission(
    predictions: Mapping[str, Mapping[str, float]] | pd.DataFrame,
    exam_ids: Iterable[str] | None = None,
    fill_value: float | None = None,
) -> pd.DataFrame:
    """Assemble a submission frame from per-exam label probabilities.

    predictions may be a mapping of exam id to label probabilities, or a frame
    already carrying the exam id either as a column or as the index. When
    exam_ids is supplied it fixes both the row set and the row order; exams
    absent from predictions are filled with fill_value if one is given, and
    raise otherwise. Silently dropping an exam would produce a submission the
    scorer rejects, so it is never allowed.
    """
    if isinstance(predictions, pd.DataFrame):
        frame = predictions.copy()
        if config.EXAM_ID_COLUMN in frame.columns:
            frame = frame.set_index(config.EXAM_ID_COLUMN, verify_integrity=True)
    else:
        frame = pd.DataFrame.from_dict(dict(predictions), orient="index")
    frame.index = frame.index.astype(str)

    missing_columns = sorted(set(config.TARGET_LABELS) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"predictions is missing label columns: {missing_columns!r}")
    frame = frame.loc[:, list(config.TARGET_LABELS)].astype(float)

    if exam_ids is not None:
        wanted = pd.Index([str(value) for value in exam_ids])
        if not wanted.is_unique:
            raise ValueError("exam_ids must not contain duplicates.")
        absent = wanted.difference(frame.index)
        if len(absent) and fill_value is None:
            raise ValueError(
                f"{len(absent)} exam(s) have no prediction and no fill_value was given; "
                f"first missing: {absent[0]!r}"
            )
        frame = frame.reindex(wanted)
        if len(absent):
            frame = frame.fillna(float(fill_value))

    frame = frame.reset_index().rename(columns={"index": config.EXAM_ID_COLUMN})
    return frame.loc[:, list(SUBMISSION_COLUMNS)]


def validate_submission(
    frame: pd.DataFrame,
    expected_exam_ids: Iterable[str] | None = None,
    sample_submission: pd.DataFrame | Path | str | None = None,
) -> pd.DataFrame:
    """Raise if the frame would be rejected by the scorer. Returns it unchanged.

    Checks the column set and its order, uniqueness of exam ids, absence of
    missing values, and that every probability is finite and within [0, 1].
    When expected_exam_ids or a sample submission is supplied, the row set is
    checked against it as well.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")

    actual = tuple(frame.columns)
    if actual != SUBMISSION_COLUMNS:
        raise ValueError(
            "Submission columns do not match the required contract.\n"
            f"  expected: {list(SUBMISSION_COLUMNS)}\n"
            f"  actual:   {list(actual)}"
        )

    ids = frame[config.EXAM_ID_COLUMN]
    if ids.isna().any():
        raise ValueError("Submission contains a missing exam id.")
    if ids.duplicated().any():
        repeated = sorted(ids[ids.duplicated()].unique())[:5]
        raise ValueError(f"Submission contains duplicate exam ids, e.g. {repeated!r}")

    values = frame.loc[:, list(config.TARGET_LABELS)]
    if not np.issubdtype(values.to_numpy().dtype, np.number):
        raise ValueError("Label columns must be numeric.")
    numbers = values.to_numpy(dtype=float)
    if not np.isfinite(numbers).all():
        row, column = np.argwhere(~np.isfinite(numbers))[0]
        raise ValueError(
            f"Non-finite probability at exam {ids.iloc[int(row)]!r}, "
            f"label {config.TARGET_LABELS[int(column)]!r}"
        )
    if numbers.min() < 0.0 or numbers.max() > 1.0:
        raise ValueError(
            f"Probabilities must lie in [0, 1]; observed range "
            f"[{numbers.min():.4f}, {numbers.max():.4f}]"
        )

    if sample_submission is not None and expected_exam_ids is None:
        if not isinstance(sample_submission, pd.DataFrame):
            sample_submission = pd.read_csv(sample_submission)
        expected_exam_ids = sample_submission[config.EXAM_ID_COLUMN]

    if expected_exam_ids is not None:
        expected = pd.Index([str(value) for value in expected_exam_ids])
        observed = pd.Index(ids.astype(str))
        missing = expected.difference(observed)
        extra = observed.difference(expected)
        if len(missing) or len(extra):
            raise ValueError(
                f"Submission row set does not match the expected exams: "
                f"{len(missing)} missing, {len(extra)} unexpected."
            )
    return frame


def write_submission(
    frame: pd.DataFrame,
    path: Path | str = "submission.csv",
    validate: bool = True,
    expected_exam_ids: Iterable[str] | None = None,
) -> Path:
    """Validate and write the submission. Validation is on by default."""
    if validate:
        validate_submission(frame, expected_exam_ids=expected_exam_ids)
    target = Path(path)
    if target.parent != Path(""):
        target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    return target


def label_priors(labels: object) -> dict[str, float]:
    """Per-label positive rate over the observed expert labels.

    Accepts a Labels namedtuple or any frame carrying the twelve label columns.
    Used as a fallback prediction for exams whose images cannot be read; a
    prior is a better guess than an arbitrary constant.
    """
    if isinstance(labels, pd.DataFrame):
        values = labels
    else:
        # A Labels namedtuple exposes .values as a frame. Note that a DataFrame
        # also has a .values attribute holding a NumPy array, so the isinstance
        # check above has to come first.
        values = getattr(labels, "values", None)
    if not isinstance(values, pd.DataFrame):
        raise TypeError("labels must expose a DataFrame of label values.")
    missing = sorted(set(config.TARGET_LABELS) - set(values.columns))
    if missing:
        raise ValueError(f"labels is missing target columns: {missing!r}")
    frame = values.loc[:, list(config.TARGET_LABELS)].astype(float)
    priors: dict[str, float] = {}
    for label in config.TARGET_LABELS:
        column = frame[label].dropna()
        priors[label] = float(column.mean()) if len(column) else 0.5
    return priors
