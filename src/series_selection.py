"""Deterministic, metadata-only series routing for the image pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import config

SLOT_DEFINITIONS: dict[str, dict[str, Any]] = {
    "sag_fluid": {"plane": "sagittal", "fluid_sensitive": 1, "prefer_fat_suppression": True},
    "cor_fluid": {"plane": "coronal", "fluid_sensitive": 1, "prefer_fat_suppression": True},
    "ax_fluid": {"plane": "axial", "fluid_sensitive": 1, "prefer_fat_suppression": True},
    "sag_t1": {"plane": "sagittal", "fluid_sensitive": 0, "prefer_fat_suppression": False},
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
    if frame[[config.EXAM_ID_COLUMN, config.SERIES_ID_COLUMN]].isna().any().any():
        raise ValueError("exam and series identifiers must not be missing.")
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
        frame["_slice_count"] = slices
    else:
        # train_series.csv has no slice-count field. NaN makes that fact
        # explicit: all candidates tie at this criterion and UID decides.
        frame["_slice_count"] = np.nan
    frame["_fat_preference"] = np.where(
        frame[f"_{config.FLUID_SENSITIVE_COLUMN}"] == 1,
        frame[f"_{config.FAT_SUPPRESSION_COLUMN}"],
        0,
    )
    return frame


def _sort_prepared(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort once so every slot can take its first matching row."""
    return frame.sort_values(
        by=[
            config.EXAM_ID_COLUMN,
            "_plane",
            f"_{config.FLUID_SENSITIVE_COLUMN}",
            "_fat_preference",
            "_slice_count",
            config.SERIES_ID_COLUMN,
        ],
        ascending=[True, True, False, False, False, True],
        na_position="last",
        kind="mergesort",
    )


def _select_prepared(frame: pd.DataFrame, exam_id: str) -> dict[str, str | None]:
    """Select all canonical slots from one already sorted frame."""
    exam_rows = frame.loc[frame[config.EXAM_ID_COLUMN] == str(exam_id)]
    selected: dict[str, str | None] = {}
    for slot, definition in SLOT_DEFINITIONS.items():
        candidates = exam_rows.loc[
            (exam_rows["_plane"] == definition["plane"])
            & (exam_rows[f"_{config.FLUID_SENSITIVE_COLUMN}"] == definition["fluid_sensitive"])
        ]
        selected[slot] = (
            str(candidates.iloc[0][config.SERIES_ID_COLUMN])
            if not candidates.empty
            else None
        )
    return selected


def select_series(series_frame: pd.DataFrame, exam_id: str) -> dict[str, str | None]:
    """Select one deterministic series ID, or None, for every canonical slot.

    Fluid-sensitive slots prefer fat suppression, then the greatest optional
    slice_count, then the lexicographically smallest SeriesInstanceUID. The
    verified train_series.csv has no slice_count column; when it is absent,
    every candidate ties at that step and UID ordering decides.
    """
    prepared = _sort_prepared(_prepare_series_frame(series_frame))
    return _select_prepared(prepared, str(exam_id))


def select_series_map(
    series_frame: pd.DataFrame,
    exam_ids: list[str] | tuple[str, ...] | pd.Index | None = None,
) -> dict[str, dict[str, str | None]]:
    """Select canonical slots for many exams while sorting metadata once."""
    prepared = _sort_prepared(_prepare_series_frame(series_frame))
    if exam_ids is None:
        ids = pd.Index(prepared[config.EXAM_ID_COLUMN].drop_duplicates().astype(str))
    else:
        ids = pd.Index([str(exam_id) for exam_id in exam_ids]).drop_duplicates()
    return {exam_id: _select_prepared(prepared, exam_id) for exam_id in ids}


def slot_coverage(
    series_frame: pd.DataFrame,
    exam_ids: list[str] | tuple[str, ...] | pd.Index | None = None,
) -> pd.DataFrame:
    """Report slot coverage using vectorized predicates, without tie-breaking."""
    frame = _prepare_series_frame(series_frame)
    if exam_ids is None:
        ids = pd.Index(frame[config.EXAM_ID_COLUMN].drop_duplicates().astype(str))
    else:
        ids = pd.Index([str(exam_id) for exam_id in exam_ids]).drop_duplicates()

    n_exams = len(ids)
    exam_key = frame[config.EXAM_ID_COLUMN]
    rows: list[dict[str, object]] = []
    for slot, definition in SLOT_DEFINITIONS.items():
        matches = (
            (frame["_plane"] == definition["plane"])
            & (frame[f"_{config.FLUID_SENSITIVE_COLUMN}"] == definition["fluid_sensitive"])
        )
        selected_by_exam = matches.groupby(exam_key, sort=False).any()
        selected_count = int(selected_by_exam.reindex(ids, fill_value=False).sum())
        rows.append({
            "slot": slot,
            "n_exams": n_exams,
            "n_selected": selected_count,
            "n_missing": n_exams - selected_count,
            "coverage": selected_count / n_exams if n_exams else float("nan"),
        })
    return pd.DataFrame(rows, columns=["slot", "n_exams", "n_selected", "n_missing", "coverage"])
