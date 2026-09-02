"""Deterministic, metadata-only series routing for the image pipeline.

SeriesDescription and scanner metadata are intentionally not consumed here.
The selector uses only the verified train_series.csv routing columns and an
optional caller-provided slice_count column for the documented tie-break.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from . import config

SLOT_DEFINITIONS: dict[str, dict[str, Any]] = {
    "sag_fluid": {
        "plane": "sagittal",
        "fluid_sensitive": 1,
        "prefer_fat_suppression": True,
    },
    "cor_fluid": {
        "plane": "coronal",
        "fluid_sensitive": 1,
        "prefer_fat_suppression": True,
    },
    "ax_fluid": {
        "plane": "axial",
        "fluid_sensitive": 1,
        "prefer_fat_suppression": True,
    },
    "sag_t1": {
        "plane": "sagittal",
        "fluid_sensitive": 0,
        "prefer_fat_suppression": False,
    },
}
SLOT_NAMES: tuple[str, ...] = tuple(SLOT_DEFINITIONS)
SLICE_COUNT_COLUMN = "slice_count"


def _prepare_series_frame(series_frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(series_frame, pd.DataFrame):
        raise TypeError("series_frame must be a pandas DataFrame.")
    required = {
        config.EXAM_ID_COLUMN,
        config.SERIES_ID_COLUMN,
        config.FLUID_SENSITIVE_COLUMN,
        config.FAT_SUPPRESSION_COLUMN,
        config.ANATOMICAL_PLANE_COLUMN,
    }
    missing = sorted(required - set(series_frame.columns))
    if missing:
        raise ValueError(f"series_frame is missing required columns: {missing!r}")

    frame = series_frame.copy()
    frame[config.EXAM_ID_COLUMN] = frame[config.EXAM_ID_COLUMN].astype(str)
    frame[config.SERIES_ID_COLUMN] = frame[config.SERIES_ID_COLUMN].astype(str)
    if frame[[config.EXAM_ID_COLUMN, config.SERIES_ID_COLUMN]].duplicated().any():
        raise ValueError("series_frame contains duplicate exam/series pairs.")

    frame["_plane"] = frame[config.ANATOMICAL_PLANE_COLUMN].astype("string").str.strip().str.casefold()
    for column in (config.FLUID_SENSITIVE_COLUMN, config.FAT_SUPPRESSION_COLUMN):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not values.isin([0, 1]).all():
            raise ValueError(f"{column} must contain only 0 or 1 values.")
        frame[f"_{column}"] = values.astype(int)

    if SLICE_COUNT_COLUMN in frame.columns:
        slices = pd.to_numeric(frame[SLICE_COUNT_COLUMN], errors="coerce")
        if slices.notna().any() and (slices.dropna() < 0).any():
            raise ValueError(f"{SLICE_COUNT_COLUMN} must be non-negative.")
        frame["_slice_count"] = slices.fillna(-1.0)
    else:
        # The verified CSV has no slice-count field. A caller may add this
        # derived column; absent it, every candidate ties at this stage.
        frame["_slice_count"] = 0.0
    return frame


def _select_prepared(frame: pd.DataFrame, exam_id: str) -> dict[str, str | None]:
    exam_rows = frame[frame[config.EXAM_ID_COLUMN] == str(exam_id)]
    selected: dict[str, str | None] = {}
    for slot, definition in SLOT_DEFINITIONS.items():
        candidates = exam_rows[
            (exam_rows["_plane"] == definition["plane"])
            & (exam_rows[f"_{config.FLUID_SENSITIVE_COLUMN}"] == definition["fluid_sensitive"])
        ].copy()
        if candidates.empty:
            selected[slot] = None
            continue

        candidates["_fat_preference"] = (
            candidates[f"_{config.FAT_SUPPRESSION_COLUMN}"]
            if definition["prefer_fat_suppression"]
            else 0
        )
        candidates = candidates.sort_values(
            by=["_fat_preference", "_slice_count", config.SERIES_ID_COLUMN],
            ascending=[False, False, True],
            kind="mergesort",
        )
        selected[slot] = str(candidates.iloc[0][config.SERIES_ID_COLUMN])
    return selected


def select_series(series_frame: pd.DataFrame, exam_id: str) -> dict[str, str | None]:
    """Select one deterministic series ID, or None, for every canonical slot.

    Fluid-sensitive slots prefer Fat_Suppression=1, then the greatest optional
    slice_count, then the lexicographically smallest SeriesInstanceUID. The T1
    slot skips the fat-suppression preference and uses slice count then UID.
    """
    frame = _prepare_series_frame(series_frame)
    return _select_prepared(frame, str(exam_id))


def slot_coverage(
    series_frame: pd.DataFrame,
    exam_ids: list[str] | tuple[str, ...] | pd.Index | None = None,
) -> pd.DataFrame:
    """Report filled and missing canonical slots across the supplied exams.

    This reads only the small series metadata frame; it does not open DICOM
    files. Pass all exam IDs explicitly when measuring a corpus that may contain
    exams with no row in the series metadata.
    """
    frame = _prepare_series_frame(series_frame)
    if exam_ids is None:
        ids = pd.Index(frame[config.EXAM_ID_COLUMN].drop_duplicates().astype(str))
    else:
        ids = pd.Index([str(exam_id) for exam_id in exam_ids]).drop_duplicates()
    rows: list[dict[str, object]] = []
    for slot in SLOT_NAMES:
        selected_count = 0
        for exam_id in ids:
            if _select_prepared(frame, exam_id)[slot] is not None:
                selected_count += 1
        n_exams = len(ids)
        rows.append(
            {
                "slot": slot,
                "n_exams": n_exams,
                "n_selected": selected_count,
                "n_missing": n_exams - selected_count,
                "coverage": selected_count / n_exams if n_exams else float("nan"),
            }
        )
    return pd.DataFrame(rows, columns=["slot", "n_exams", "n_selected", "n_missing", "coverage"])
