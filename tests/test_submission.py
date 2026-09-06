"""Submission contract tests. No competition data required."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.submission import (
    SUBMISSION_COLUMNS,
    build_submission,
    label_priors,
    validate_submission,
    write_submission,
)

LABELS = list(config.TARGET_LABELS)
EXAM = config.EXAM_ID_COLUMN


def _predictions(exam_ids, value=0.5):
    return {exam_id: {label: value for label in LABELS} for exam_id in exam_ids}


def test_columns_are_exact_and_ordered():
    frame = build_submission(_predictions(["a", "b"]))
    assert tuple(frame.columns) == SUBMISSION_COLUMNS
    assert SUBMISSION_COLUMNS[0] == EXAM
    assert list(SUBMISSION_COLUMNS[1:]) == LABELS


def test_exam_ids_and_order_follow_the_request():
    frame = build_submission(_predictions(["b", "a", "c"]), exam_ids=["c", "a", "b"])
    assert frame[EXAM].tolist() == ["c", "a", "b"]


def test_missing_exam_raises_without_fill_value():
    with pytest.raises(ValueError, match="no prediction"):
        build_submission(_predictions(["a"]), exam_ids=["a", "b"])


def test_missing_exam_is_filled_when_allowed():
    frame = build_submission(_predictions(["a"]), exam_ids=["a", "b"], fill_value=0.25)
    assert frame.loc[frame[EXAM] == "b", LABELS].to_numpy().tolist() == [[0.25] * 12]


def test_accepts_a_frame_with_the_exam_id_as_index():
    source = pd.DataFrame(0.4, index=pd.Index(["a", "b"], name=EXAM), columns=LABELS)
    frame = build_submission(source)
    assert frame.shape == (2, 13)


def test_validate_rejects_wrong_column_order():
    frame = build_submission(_predictions(["a"]))
    shuffled = frame.loc[:, [EXAM] + LABELS[::-1]]
    with pytest.raises(ValueError, match="do not match"):
        validate_submission(shuffled)


def test_validate_rejects_duplicate_exam_ids():
    frame = build_submission(_predictions(["a"]))
    doubled = pd.concat([frame, frame], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_submission(doubled)


def test_validate_rejects_out_of_range_probability():
    frame = build_submission(_predictions(["a"]))
    frame.loc[0, "ACL"] = 1.4
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_submission(frame)


def test_validate_rejects_nan():
    frame = build_submission(_predictions(["a"]))
    frame.loc[0, "MCL"] = np.nan
    with pytest.raises(ValueError, match="Non-finite"):
        validate_submission(frame)


def test_validate_rejects_a_mismatched_row_set():
    frame = build_submission(_predictions(["a", "b"]))
    with pytest.raises(ValueError, match="row set"):
        validate_submission(frame, expected_exam_ids=["a", "b", "c"])


def test_validate_accepts_a_correct_submission():
    frame = build_submission(_predictions(["a", "b"]))
    assert validate_submission(frame, expected_exam_ids=["a", "b"]) is frame


def test_write_round_trips(tmp_path):
    frame = build_submission(_predictions(["a", "b"]))
    path = write_submission(frame, tmp_path / "submission.csv")
    reloaded = pd.read_csv(path)
    reloaded[EXAM] = reloaded[EXAM].astype(str)
    validate_submission(reloaded, expected_exam_ids=["a", "b"])


def test_label_priors_ignore_missing_values():
    values = pd.DataFrame(np.nan, index=["a", "b", "c"], columns=LABELS)
    values.loc[["a", "b"], "ACL"] = [1.0, 0.0]
    priors = label_priors(values)
    assert priors["ACL"] == pytest.approx(0.5)
    assert priors["MCL"] == pytest.approx(0.5)  # no observations, defaults to 0.5
