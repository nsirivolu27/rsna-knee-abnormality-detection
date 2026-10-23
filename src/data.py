"""Build a lightweight exam index from verified tables and folders."""

from pathlib import Path
from typing import Optional, Union
import warnings

import pandas as pd

from . import config


DataPath = Union[Path, str]


def build_exam_index(
    train_csv: Optional[DataPath] = None,
    train_series_csv: Optional[DataPath] = None,
    series_root: Optional[DataPath] = None,
) -> pd.DataFrame:
    """Join train labels, series metadata, and per-series folder status.

    The function never opens DICOM files. If the DICOM root is absent, it
    returns a clearly marked CSV-only index instead of failing.
    """
    train_path = Path(train_csv) if train_csv is not None else config.TRAIN_CSV
    series_csv_path = (
        Path(train_series_csv)
        if train_series_csv is not None
        else config.TRAIN_SERIES_CSV
    )
    root = Path(series_root) if series_root is not None else config.TRAIN_SERIES_ROOT

    train = pd.read_csv(train_path)
    train = train.set_index(config.EXAM_ID_COLUMN, verify_integrity=True)

    series = pd.read_csv(
        series_csv_path,
        usecols=[
            config.EXAM_ID_COLUMN,
            config.SERIES_ID_COLUMN,
            config.FLUID_SENSITIVE_COLUMN,
            config.FAT_SUPPRESSION_COLUMN,
            config.ANATOMICAL_PLANE_COLUMN,
        ],
    )
    series["series_directory"] = series.apply(
        lambda row: root / str(row[config.EXAM_ID_COLUMN]) / str(row[config.SERIES_ID_COLUMN]),
        axis=1,
    )

    if root.is_dir():
        series["series_directory_exists"] = series["series_directory"].map(
            lambda path: path.is_dir()
        )
        dicom_status = "DICOM directory checks performed without reading pixels."
    else:
        warnings.warn(
            "DICOM root is absent; returning a CSV-only exam index.",
            RuntimeWarning,
            stacklevel=2,
        )
        series["series_directory_exists"] = False
        dicom_status = "CSV-only: DICOM root is absent."

    series_summary = (
        series.groupby(config.EXAM_ID_COLUMN, sort=False)
        .agg(
            series_count=(config.SERIES_ID_COLUMN, "nunique"),
            fluid_sensitive_series_count=(config.FLUID_SENSITIVE_COLUMN, "sum"),
            fat_suppressed_series_count=(config.FAT_SUPPRESSION_COLUMN, "sum"),
            series_directories_present=("series_directory_exists", "sum"),
        )
    )
    series_summary["dicom_root_present"] = root.is_dir()
    series_summary["dicom_status"] = dicom_status

    exam_index = train.join(series_summary, how="left", validate="one_to_one")
    exam_index["series_count"] = exam_index["series_count"].fillna(0).astype(int)
    exam_index["series_directories_present"] = (
        exam_index["series_directories_present"].fillna(0).astype(int)
    )
    return exam_index.reset_index()
