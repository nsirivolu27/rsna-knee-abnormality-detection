"""Lazy, per-series DICOM loading and metadata sampling."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import pydicom

from . import config


SeriesPath = Union[Path, str]


@dataclass
class SeriesLoadResult:
    """Result of loading one DICOM series, including non-fatal warnings."""

    volume: Optional[np.ndarray]
    metadata: dict[str, object]
    warnings: list[dict[str, str]]


def _warning(code: str, message: str, path: Optional[Path] = None) -> dict[str, str]:
    result = {"code": code, "message": message}
    if path is not None:
        result["path"] = str(path)
    return result


def _series_files(series_path: Path) -> list[Path]:
    """List files in one supplied series directory, never a whole DICOM tree."""
    try:
        children = sorted(series_path.iterdir())
    except OSError:
        return []
    return [
        child
        for child in children
        if child.is_file() and not child.name.startswith(".")
    ]


def _slice_sort_key(dataset: pydicom.dataset.Dataset, path: Path) -> tuple[object, ...]:
    position = getattr(dataset, "ImagePositionPatient", None)
    if position is not None and len(position) >= 3:
        try:
            return (0, float(position[2]), int(getattr(dataset, "InstanceNumber", 0)), path.name)
        except (TypeError, ValueError):
            pass

    try:
        instance_number = int(getattr(dataset, "InstanceNumber", 0))
    except (TypeError, ValueError):
        instance_number = 0
    return (1, instance_number, path.name)


def _plain_value(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def extract_metadata(
    dataset: pydicom.dataset.Dataset,
    volume: Optional[np.ndarray] = None,
) -> dict[str, object]:
    """Extract the DICOM fields needed for protocol and site analysis."""
    metadata = {
        field: _plain_value(getattr(dataset, field, None))
        for field in config.DICOM_METADATA_FIELDS
    }
    metadata["shape"] = tuple(int(size) for size in volume.shape) if volume is not None else None
    metadata["dtype"] = str(volume.dtype) if volume is not None else None
    return metadata


def load_series(
    series_path: SeriesPath,
    read_pixels: bool = True,
) -> SeriesLoadResult:
    """Read one supplied series into an ordered volume without raising on bad input."""
    path = Path(series_path)
    warnings: list[dict[str, str]] = []

    if not path.exists():
        warnings.append(_warning("missing_series", "Series directory does not exist.", path))
        return SeriesLoadResult(None, {}, warnings)
    if not path.is_dir():
        warnings.append(_warning("not_a_directory", "Series path is not a directory.", path))
        return SeriesLoadResult(None, {}, warnings)

    files = _series_files(path)
    if not files:
        warnings.append(_warning("empty_series", "Series directory contains no files.", path))
        return SeriesLoadResult(None, {}, warnings)

    datasets: list[tuple[Path, pydicom.dataset.Dataset]] = []
    for file_path in files:
        try:
            dataset = pydicom.dcmread(
                str(file_path),
                force=True,
                stop_before_pixels=not read_pixels,
            )
            datasets.append((file_path, dataset))
        except Exception as error:
            warnings.append(
                _warning(
                    "malformed_dicom",
                    f"Could not read DICOM file: {error}",
                    file_path,
                )
            )

    if not datasets:
        warnings.append(_warning("no_readable_slices", "No readable DICOM files were found.", path))
        return SeriesLoadResult(None, {}, warnings)

    datasets.sort(key=lambda item: _slice_sort_key(item[1], item[0]))
    metadata = extract_metadata(datasets[0][1])

    if not read_pixels:
        return SeriesLoadResult(None, metadata, warnings)

    arrays: list[np.ndarray] = []
    expected_shape: Optional[tuple[int, ...]] = None
    for file_path, dataset in datasets:
        try:
            array = np.asarray(dataset.pixel_array)
        except Exception as error:
            warnings.append(
                _warning(
                    "malformed_pixels",
                    f"Could not decode pixel data: {error}",
                    file_path,
                )
            )
            continue

        if expected_shape is None:
            expected_shape = tuple(int(size) for size in array.shape)
        if tuple(array.shape) != expected_shape:
            warnings.append(
                _warning(
                    "shape_mismatch",
                    f"Expected slice shape {expected_shape}, got {tuple(array.shape)}.",
                    file_path,
                )
            )
            continue
        arrays.append(array)

    if not arrays:
        warnings.append(_warning("no_decodable_pixels", "No compatible pixel arrays were decoded.", path))
        return SeriesLoadResult(None, metadata, warnings)

    volume = np.stack(arrays, axis=0)
    metadata = extract_metadata(datasets[0][1], volume=volume)
    return SeriesLoadResult(volume, metadata, warnings)


def _first_series_file(series_path: Path) -> Optional[Path]:
    """Return one file from one series directory without scanning its siblings."""
    try:
        for child in sorted(series_path.iterdir()):
            if child.is_file() and not child.name.startswith("."):
                return child
    except OSError:
        return None
    return None


def sample_series_metadata(
    n_exams: int = 200,
    seed: int = config.RANDOM_SEED,
    series_csv: Optional[SeriesPath] = None,
    series_root: Optional[SeriesPath] = None,
) -> pd.DataFrame:
    """Read one DICOM file per series for a random sample of exams.

    The sample is selected from the small series CSV, then only one file in
    each selected series directory is read. It never scans all 24,371 series.
    """
    csv_path = Path(series_csv) if series_csv is not None else config.TRAIN_SERIES_CSV
    root = Path(series_root) if series_root is not None else config.TRAIN_SERIES_ROOT
    series_frame = pd.read_csv(
        csv_path,
        usecols=[
            config.EXAM_ID_COLUMN,
            config.SERIES_ID_COLUMN,
            config.FLUID_SENSITIVE_COLUMN,
            config.FAT_SUPPRESSION_COLUMN,
            config.ANATOMICAL_PLANE_COLUMN,
        ],
    )

    exam_ids = series_frame[config.EXAM_ID_COLUMN].drop_duplicates()
    sample_size = min(max(n_exams, 0), len(exam_ids))
    if sample_size == 0:
        return pd.DataFrame()

    sampled_exams = set(exam_ids.sample(n=sample_size, random_state=seed))
    sampled_series = series_frame[
        series_frame[config.EXAM_ID_COLUMN].isin(sampled_exams)
    ]

    rows: list[dict[str, object]] = []
    metadata_columns = list(config.DICOM_METADATA_FIELDS) + ["shape", "dtype"]
    for row in sampled_series.itertuples(index=False):
        study_id = getattr(row, config.EXAM_ID_COLUMN)
        series_id = getattr(row, config.SERIES_ID_COLUMN)
        directory = root / str(study_id) / str(series_id)
        file_path = _first_series_file(directory)
        output = {
            config.EXAM_ID_COLUMN: study_id,
            config.SERIES_ID_COLUMN: series_id,
            config.FLUID_SENSITIVE_COLUMN: getattr(row, config.FLUID_SENSITIVE_COLUMN),
            config.FAT_SUPPRESSION_COLUMN: getattr(row, config.FAT_SUPPRESSION_COLUMN),
            config.ANATOMICAL_PLANE_COLUMN: getattr(row, config.ANATOMICAL_PLANE_COLUMN),
            "warning_code": None,
            "warning_message": None,
        }
        output.update({column: None for column in metadata_columns})

        if file_path is None:
            output["warning_code"] = "missing_series_file"
            output["warning_message"] = "No readable candidate file was found."
            rows.append(output)
            continue

        try:
            dataset = pydicom.dcmread(str(file_path), force=True, stop_before_pixels=True)
            output.update(extract_metadata(dataset))
        except Exception as error:
            output["warning_code"] = "malformed_dicom"
            output["warning_message"] = str(error)
        rows.append(output)

    return pd.DataFrame(rows)



def _requested_exam_ids(exams: object) -> list[str] | None:
    """Normalize an optional exam selection without scanning any DICOM tree."""
    if exams is None:
        return None
    if isinstance(exams, pd.DataFrame):
        if config.EXAM_ID_COLUMN not in exams.columns:
            raise ValueError(f"exams must contain {config.EXAM_ID_COLUMN!r}.")
        values = exams[config.EXAM_ID_COLUMN].tolist()
    elif isinstance(exams, (str, Path)):
        values = [exams]
    else:
        values = list(exams)  # type: ignore[arg-type]
    return list(dict.fromkeys(str(value) for value in values if value is not None))


def _metadata_output_columns() -> list[str]:
    return [
        config.EXAM_ID_COLUMN,
        "selected_series_uid",
        "selection_strategy",
        *config.DICOM_METADATA_FIELDS,
        "warning_code",
        "warning_message",
    ]


def exam_metadata(
    exams: object = None,
    series_csv: Optional[SeriesPath] = None,
    series_root: Optional[SeriesPath] = None,
    cache_path: Optional[SeriesPath] = None,
) -> pd.DataFrame:
    """Read one DICOM header per exam and return full-corpus metadata.

    A sagittal series is preferred for the single header read; the first
    available series is used as a deterministic fallback. The function reads
    the series table once and only enters the selected exam/series directory,
    so it performs roughly one DICOM read per exam rather than one per series.
    DICOM failures become structured warning columns and never abort the
    remaining exams. An existing parquet cache is returned unchanged.
    """
    if cache_path is not None:
        cache = Path(cache_path)
        if cache.exists():
            return pd.read_parquet(cache)

    csv_path = Path(series_csv) if series_csv is not None else config.TRAIN_SERIES_CSV
    root = Path(series_root) if series_root is not None else config.TRAIN_SERIES_ROOT
    series_frame = pd.read_csv(
        csv_path,
        usecols=[
            config.EXAM_ID_COLUMN,
            config.SERIES_ID_COLUMN,
            config.ANATOMICAL_PLANE_COLUMN,
        ],
        dtype={
            config.EXAM_ID_COLUMN: "string",
            config.SERIES_ID_COLUMN: "string",
            config.ANATOMICAL_PLANE_COLUMN: "string",
        },
    )
    series_frame[config.EXAM_ID_COLUMN] = series_frame[config.EXAM_ID_COLUMN].astype(str)
    series_frame[config.SERIES_ID_COLUMN] = series_frame[config.SERIES_ID_COLUMN].astype(str)

    requested = _requested_exam_ids(exams)
    exam_ids = (
        requested
        if requested is not None
        else series_frame[config.EXAM_ID_COLUMN].drop_duplicates().tolist()
    )
    by_exam = {
        str(exam_id): group
        for exam_id, group in series_frame.groupby(config.EXAM_ID_COLUMN, sort=False)
    }

    rows: list[dict[str, object]] = []
    for exam_id in exam_ids:
        output: dict[str, object] = {
            column: None for column in _metadata_output_columns()
        }
        output[config.EXAM_ID_COLUMN] = str(exam_id)
        output["warning_code"] = None
        output["warning_message"] = None

        group = by_exam.get(str(exam_id))
        if group is None or group.empty:
            output["warning_code"] = "missing_exam_series"
            output["warning_message"] = "Exam was not present in the series table."
            rows.append(output)
            continue

        sagittal = group[
            group[config.ANATOMICAL_PLANE_COLUMN]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.casefold()
            .isin({"sagittal", "sag"})
        ]
        selected = sagittal.iloc[0] if not sagittal.empty else group.iloc[0]
        strategy = "sagittal" if not sagittal.empty else "first_available"
        series_id = str(selected[config.SERIES_ID_COLUMN])
        output["selected_series_uid"] = series_id
        output["selection_strategy"] = strategy
        series_path = root / str(exam_id) / series_id
        file_path = _first_series_file(series_path)
        if file_path is None:
            output["warning_code"] = "missing_series_file"
            output["warning_message"] = "No readable candidate file was found."
            rows.append(output)
            continue

        try:
            dataset = pydicom.dcmread(
                str(file_path), force=True, stop_before_pixels=True
            )
            output.update(extract_metadata(dataset))
        except Exception as error:
            output["warning_code"] = "malformed_dicom"
            output["warning_message"] = str(error)
        rows.append(output)

    result = pd.DataFrame(rows, columns=_metadata_output_columns())
    if cache_path is not None:
        cache = Path(cache_path)
        cache.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(cache, index=False)
    return result
