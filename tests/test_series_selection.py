import pandas as pd
import time
import pytest

from src import config
from src.series_selection import select_series, slot_coverage


def _series_frame():
    return pd.DataFrame(
        {
            config.EXAM_ID_COLUMN: [
                "e1", "e1", "e1", "e1", "e1", "e1", "e1", "e2", "e2", "e2"
            ],
            config.SERIES_ID_COLUMN: [
                "sag0", "sag1", "cor0", "cor1", "ax0", "t1b", "t1a", "sag2", "sag3", "ax2"
            ],
            config.FLUID_SENSITIVE_COLUMN: [1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
            config.FAT_SUPPRESSION_COLUMN: [0, 1, 1, 1, 1, 0, 0, 0, 0, 1],
            config.ANATOMICAL_PLANE_COLUMN: [
                "Sagittal", "Sagittal", "Coronal", "Coronal", "Axial", "Sagittal", "Sagittal", "Sagittal", "Sagittal", "Axial"
            ],
            "slice_count": [30, 20, 20, 22, 16, 10, 12, 18, 18, 14],
        }
    )


def test_select_series_applies_slot_rules_and_ties():
    selected = select_series(_series_frame(), "e1")

    assert selected == {
        "sag_fluid": "sag1",
        "cor_fluid": "cor1",
        "ax_fluid": "ax0",
        "sag_t1": "t1a",
    }

    tied = _series_frame().iloc[[7, 8]].copy()
    tied["slice_count"] = 18
    assert select_series(tied, "e2")["sag_fluid"] == "sag2"

    no_slice_counts = tied.drop(columns=["slice_count"])
    assert select_series(no_slice_counts, "e2")["sag_fluid"] == "sag2"


def test_select_series_returns_none_for_missing_slots():
    selected = select_series(_series_frame(), "missing")
    assert selected == {slot: None for slot in ("sag_fluid", "cor_fluid", "ax_fluid", "sag_t1")}


def test_slot_coverage_counts_missing_requested_exams():
    coverage = slot_coverage(_series_frame(), exam_ids=["e1", "e2", "e3"])
    values = coverage.set_index("slot")

    assert values.loc["sag_fluid", "n_selected"] == 2
    assert values.loc["cor_fluid", "n_selected"] == 1
    assert values.loc["ax_fluid", "n_selected"] == 2
    assert values.loc["sag_t1", "n_selected"] == 1
    assert values.loc["sag_fluid", "coverage"] == pytest.approx(2 / 3)


def test_series_selection_rejects_duplicate_pairs():
    frame = _series_frame()
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate exam/series"):
        select_series(frame, "e1")


def test_slot_coverage_scales_to_5000_exams():
    rows = []
    for index in range(5000):
        exam_id = f"exam-{index:05d}"
        rows.extend(
            [
                [exam_id, f"{exam_id}-sag", 1, 1, "Sagittal"],
                [exam_id, f"{exam_id}-cor", 1, 0, "Coronal"],
                [exam_id, f"{exam_id}-ax", 1, 0, "Axial"],
                [exam_id, f"{exam_id}-t1", 0, 0, "Sagittal"],
            ]
        )
    frame = pd.DataFrame(
        rows,
        columns=[
            config.EXAM_ID_COLUMN,
            config.SERIES_ID_COLUMN,
            config.FLUID_SENSITIVE_COLUMN,
            config.FAT_SUPPRESSION_COLUMN,
            config.ANATOMICAL_PLANE_COLUMN,
        ],
    )

    started = time.perf_counter()
    coverage = slot_coverage(frame)
    elapsed = time.perf_counter() - started

    assert elapsed < 2.0, f"slot_coverage took {elapsed:.3f}s"
    assert coverage["n_selected"].tolist() == [5000, 5000, 5000, 5000]
