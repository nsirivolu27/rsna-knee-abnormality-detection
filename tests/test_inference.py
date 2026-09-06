"""Inference loop tests using a stub predictor and synthetic series."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.inference import (
    constant_predictor,
    predict_exams,
    prior_predictor,
    run_inference,
    summarize_run,
)
from src.submission import validate_submission

LABELS = list(config.TARGET_LABELS)
EXAM = config.EXAM_ID_COLUMN


@pytest.fixture()
def empty_series_frame():
    """A valid series table whose directories do not exist on disk."""
    rows = []
    for exam_id in ("exam-1", "exam-2", "exam-3"):
        for plane in ("Sagittal", "Coronal", "Axial"):
            rows.append(
                {
                    EXAM: exam_id,
                    config.SERIES_ID_COLUMN: f"{exam_id}-{plane.lower()}",
                    config.FLUID_SENSITIVE_COLUMN: 1,
                    config.FAT_SUPPRESSION_COLUMN: 1,
                    config.ANATOMICAL_PLANE_COLUMN: plane,
                }
            )
    return pd.DataFrame(rows)


def test_every_requested_exam_gets_a_row(empty_series_frame, tmp_path):
    ids = ["exam-1", "exam-2", "exam-3"]
    predictions, diagnostics = predict_exams(
        ids, constant_predictor(0.5),
        series_frame=empty_series_frame, series_root=tmp_path,
    )
    assert list(predictions.index) == ids
    assert list(predictions.columns) == LABELS
    assert len(diagnostics) == 3


def test_unreadable_images_are_reported_not_raised(empty_series_frame, tmp_path):
    predictions, diagnostics = predict_exams(
        ["exam-1"], constant_predictor(0.5),
        series_frame=empty_series_frame, series_root=tmp_path,
    )
    assert diagnostics.status.tolist() == ["no_images"]
    assert diagnostics.slots_present.tolist() == [0]
    assert predictions.notna().all().all()


def test_a_raising_predictor_falls_back_without_ending_the_run(empty_series_frame, tmp_path):
    def broken(_exam):
        raise RuntimeError("model exploded")

    predictions, diagnostics = predict_exams(
        ["exam-1", "exam-2"], broken,
        series_frame=empty_series_frame, series_root=tmp_path, fallback=0.3,
    )
    assert len(predictions) == 2
    assert (predictions.to_numpy() == 0.3).all()
    assert diagnostics.status.tolist() == ["failed", "failed"]
    assert "model exploded" in diagnostics.detail.iloc[0]


def test_predictor_returning_the_wrong_length_is_a_failure(empty_series_frame, tmp_path):
    _, diagnostics = predict_exams(
        ["exam-1"], lambda _exam: [0.5, 0.5],
        series_frame=empty_series_frame, series_root=tmp_path,
    )
    assert diagnostics.status.tolist() == ["failed"]
    assert "expected 12" in diagnostics.detail.iloc[0]


def test_predictor_returning_out_of_range_is_a_failure(empty_series_frame, tmp_path):
    _, diagnostics = predict_exams(
        ["exam-1"], lambda _exam: np.full(12, 3.0),
        series_frame=empty_series_frame, series_root=tmp_path,
    )
    assert diagnostics.status.tolist() == ["failed"]


def test_predictor_may_return_a_label_mapping(empty_series_frame, tmp_path):
    mapping = {label: 0.2 for label in LABELS}
    predictions, diagnostics = predict_exams(
        ["exam-1"], lambda _exam: mapping,
        series_frame=empty_series_frame, series_root=tmp_path,
    )
    assert (predictions.to_numpy() == 0.2).all()
    assert diagnostics.status.tolist() == ["no_images"]


def test_prior_predictor_rejects_a_prior_outside_the_unit_interval():
    priors = {label: 0.5 for label in LABELS}
    priors["ACL"] = 1.4
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        prior_predictor(priors)


def test_prior_predictor_uses_the_supplied_rates(empty_series_frame, tmp_path):
    priors = {label: 0.05 * (index + 1) for index, label in enumerate(LABELS)}
    predictions, _ = predict_exams(
        ["exam-1"], prior_predictor(priors),
        series_frame=empty_series_frame, series_root=tmp_path,
    )
    assert predictions.loc["exam-1", "ACL"] == pytest.approx(0.05)


def test_run_inference_writes_a_valid_submission(empty_series_frame, tmp_path):
    ids = ["exam-1", "exam-2", "exam-3"]
    output = tmp_path / "submission.csv"
    submission, diagnostics = run_inference(
        constant_predictor(0.5), exam_ids=ids, output_path=output,
        series_frame=empty_series_frame, series_root=tmp_path,
    )
    validate_submission(submission, expected_exam_ids=ids)
    reloaded = pd.read_csv(output)
    assert reloaded.shape == (3, 13)
    assert "3 exams" in summarize_run(diagnostics)


def test_duplicate_exam_ids_are_rejected(empty_series_frame, tmp_path):
    with pytest.raises(ValueError, match="duplicates"):
        predict_exams(["exam-1", "exam-1"], constant_predictor(),
                      series_frame=empty_series_frame, series_root=tmp_path)
