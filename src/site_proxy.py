"""Build conservative, non-text site proxies for grouped cross-validation.

This module uses report metadata only for language and de-identification
signatures. It never extracts pathology labels from report text.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any, Optional

import pandas as pd

from . import config
from . import reports as report_utils


SITE_PROXY_KEY_COLUMNS: tuple[str, ...] = (
    "normalized_manufacturer",
    "model",
    "MagneticFieldStrength",
    "detected_language",
    "placeholder_signature",
)


def _is_missing(value: Any) -> bool:
    """Return whether a scalar metadata value is missing."""
    if value is None or value is pd.NA:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, bool) else False


def normalize_manufacturer(value: object) -> str:
    """Collapse documented vendor spellings to a stable canonical name.

    The source spelling remains in the separate Manufacturer column. Unknown
    non-empty vendors are normalized conservatively rather than discarded.
    """
    if _is_missing(value) or not str(value).strip():
        return "unknown"

    raw = str(value).strip()
    token = re.sub(r"[^a-z0-9]+", " ", raw.casefold()).strip()
    token = re.sub(r"\s+", " ", token)

    if token in {"siemens", "siemens healthineers"}:
        return "siemens"
    if token in {"philips", "philips healthcare", "philips medical systems"}:
        return "philips"
    if token in {"gehc", "ge medical systems", "ge"}:
        return "ge"
    if token in {"toshiba", "canon mec", "canon"}:
        return "canon"

    return token.replace(" ", "_") or "unknown"


def _round_measurement(value: object, decimals: int) -> object:
    if _is_missing(value):
        return pd.NA
    try:
        return round(float(value), decimals)
    except (TypeError, ValueError):
        return value


def _representative_value(values: Iterable[object]) -> object:
    """Return the deterministic mode, preferring the lexicographically first tie."""
    valid = [value for value in values if not _is_missing(value)]
    if not valid:
        return pd.NA

    counts: dict[str, int] = {}
    first_value: dict[str, object] = {}
    for value in valid:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
        first_value.setdefault(key, value)
    winner = min(counts, key=lambda key: (-counts[key], key))
    return first_value[winner]


def _with_exam_column(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    result = frame.copy()
    if config.EXAM_ID_COLUMN not in result.columns:
        if result.index.name == config.EXAM_ID_COLUMN:
            result = result.reset_index()
        else:
            raise ValueError(f"{name} must contain {config.EXAM_ID_COLUMN!r}.")
    result[config.EXAM_ID_COLUMN] = result[config.EXAM_ID_COLUMN].astype(str)
    if result[config.EXAM_ID_COLUMN].duplicated().any() and name == "reports":
        duplicate_ids = result.loc[
            result[config.EXAM_ID_COLUMN].duplicated(keep=False), config.EXAM_ID_COLUMN
        ].drop_duplicates().tolist()
        raise ValueError(f"reports contains duplicate exam IDs: {duplicate_ids!r}")
    return result


def _source_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> Optional[str]:
    return next((column for column in candidates if column in frame.columns), None)


def _prepare_reports(reports: Optional[pd.DataFrame]) -> pd.DataFrame:
    if reports is None:
        return pd.DataFrame(
            columns=[config.EXAM_ID_COLUMN, "detected_language", "placeholder_signature"]
        )

    frame = _with_exam_column(reports, "reports")
    language_column = _source_column(
        frame, ("detected_language", "Language", "language")
    )
    signature_column = _source_column(
        frame, ("placeholder_signature", "PlaceholderSignature", "Placeholder_Signature")
    )
    report_column = config.REPORT_COLUMN if config.REPORT_COLUMN in frame.columns else None

    output = frame[[config.EXAM_ID_COLUMN]].copy()
    if language_column is not None:
        output["detected_language"] = frame[language_column].map(
            lambda value: pd.NA if _is_missing(value) else str(value).strip().lower()
        )
    elif report_column is not None:
        output["detected_language"] = frame[report_column].map(
            report_utils.detect_language
        )
    else:
        output["detected_language"] = pd.NA

    if signature_column is not None:
        output["placeholder_signature"] = frame[signature_column].map(
            lambda value: "unknown" if _is_missing(value) else str(value)
        )
    elif report_column is not None:
        output["placeholder_signature"] = frame[report_column].map(
            report_utils.placeholder_signature
        )
    else:
        output["placeholder_signature"] = "unknown"
    return output


def _prepare_series_metadata(series_metadata: Optional[pd.DataFrame]) -> pd.DataFrame:
    columns = [
        config.EXAM_ID_COLUMN,
        "Manufacturer",
        "normalized_manufacturer",
        "ManufacturerModelName",
        "model",
        "MagneticFieldStrength",
        "SliceThickness",
    ]
    if series_metadata is None:
        return pd.DataFrame(columns=columns)

    frame = _with_exam_column(series_metadata, "series_metadata")
    manufacturer_column = _source_column(frame, ("Manufacturer", "manufacturer"))
    model_column = _source_column(frame, ("ManufacturerModelName", "model", "Model"))
    field_column = _source_column(
        frame, ("MagneticFieldStrength", "magnetic_field_strength")
    )
    thickness_column = _source_column(frame, ("SliceThickness", "slice_thickness"))

    frame["MagneticFieldStrength"] = (
        frame[field_column].map(lambda value: _round_measurement(value, 3))
        if field_column is not None
        else pd.NA
    )
    frame["SliceThickness"] = (
        frame[thickness_column].map(lambda value: _round_measurement(value, 3))
        if thickness_column is not None
        else pd.NA
    )

    rows: list[dict[str, object]] = []
    for exam_id, group in frame.groupby(config.EXAM_ID_COLUMN, sort=False, dropna=False):
        raw_manufacturer = (
            _representative_value(group[manufacturer_column])
            if manufacturer_column is not None
            else pd.NA
        )
        model = (
            _representative_value(group[model_column])
            if model_column is not None
            else pd.NA
        )
        rows.append(
            {
                config.EXAM_ID_COLUMN: str(exam_id),
                "Manufacturer": raw_manufacturer,
                "normalized_manufacturer": normalize_manufacturer(raw_manufacturer),
                "ManufacturerModelName": model,
                "model": model,
                "MagneticFieldStrength": _representative_value(
                    group["MagneticFieldStrength"]
                ),
                "SliceThickness": _representative_value(group["SliceThickness"]),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _key_component(value: object) -> object:
    return "unknown" if _is_missing(value) or str(value).strip() == "" else value


def build_site_proxy(
    series_metadata: Optional[pd.DataFrame] = None,
    reports: Optional[pd.DataFrame] = None,
    *,
    min_group_size: int = 10,
) -> pd.DataFrame:
    """Build one site-proxy row per exam.

    Series metadata may contain one row per series, as returned by
    src.dicom_io.sample_series_metadata. Metadata is reduced to a deterministic
    per-exam mode. Reports may be the output of src.reports.load_reports or a
    frame containing Report.

    The five-part key is aspirational, not the grouping that usually operates.
    At the default min_group_size=10, roughly 84% of exams coarsen to
    (normalized_manufacturer, detected_language), so folds are not grouped at
    scanner granularity. Smaller groups are assigned to that two-part key, then
    to (detected_language,) if still too small. The original and assigned keys,
    group size, and assignment level are all returned for auditability.
    """
    if min_group_size < 1:
        raise ValueError("min_group_size must be at least 1.")

    metadata = _prepare_series_metadata(series_metadata)
    report_proxies = _prepare_reports(reports)
    result = metadata.merge(report_proxies, on=config.EXAM_ID_COLUMN, how="outer")

    defaults = {
        "normalized_manufacturer": "unknown",
        "model": pd.NA,
        "MagneticFieldStrength": pd.NA,
        "SliceThickness": pd.NA,
        "detected_language": pd.NA,
        "placeholder_signature": "unknown",
    }
    for column, default in defaults.items():
        if column not in result.columns:
            result[column] = default

    result["full_site_proxy_key"] = [
        (
            _key_component(row["normalized_manufacturer"]),
            _key_component(row["model"]),
            _key_component(row["MagneticFieldStrength"]),
            _key_component(row["detected_language"]),
            _key_component(row["placeholder_signature"]),
        )
        for _, row in result.iterrows()
    ]
    full_sizes = result["full_site_proxy_key"].value_counts(dropna=False)
    manufacturer_language_keys = result.apply(
        lambda row: (
            _key_component(row["normalized_manufacturer"]),
            _key_component(row["detected_language"]),
        ),
        axis=1,
    )
    language_keys = result["detected_language"].map(
        lambda value: (_key_component(value),)
    )
    manufacturer_language_sizes = manufacturer_language_keys.value_counts(dropna=False)
    language_sizes = language_keys.value_counts(dropna=False)
    small_full = result["full_site_proxy_key"].map(full_sizes) < min_group_size
    fallback_manufacturer_language = set(
        manufacturer_language_keys[small_full]
        .loc[
            manufacturer_language_keys[small_full].map(manufacturer_language_sizes)
            >= min_group_size
        ]
        .tolist()
    )
    fallback_language = set(
        language_keys[small_full]
        .loc[
            manufacturer_language_keys[small_full].map(manufacturer_language_sizes)
            < min_group_size
        ]
        .loc[language_keys[small_full].map(language_sizes) >= min_group_size]
        .tolist()
    )

    assigned_keys: list[tuple[object, ...]] = []
    assignment_levels: list[str] = []
    for full_key, manufacturer_language_key, language_key in zip(
        result["full_site_proxy_key"], manufacturer_language_keys, language_keys
    ):
        if manufacturer_language_key in fallback_manufacturer_language:
            assigned_keys.append(manufacturer_language_key)
            assignment_levels.append("manufacturer_language")
        elif language_key in fallback_language:
            assigned_keys.append(language_key)
            assignment_levels.append("language")
        elif int(full_sizes[full_key]) >= min_group_size:
            assigned_keys.append(full_key)
            assignment_levels.append("full")
        elif int(manufacturer_language_sizes[manufacturer_language_key]) >= min_group_size:
            assigned_keys.append(manufacturer_language_key)
            assignment_levels.append("manufacturer_language")
        else:
            assigned_keys.append(language_key)
            assignment_levels.append("language")

    result["site_proxy_key"] = assigned_keys
    result["site_proxy_assignment_level"] = assignment_levels
    result["site_proxy"] = result["site_proxy_key"].map(
        lambda key: "|".join(str(part) for part in key)
    )

    assigned_sizes = result["site_proxy_key"].value_counts(dropna=False)
    result["site_proxy_group_size"] = result["site_proxy_key"].map(assigned_sizes).astype(int)
    result["site_proxy_under_minimum"] = result["site_proxy_group_size"] < min_group_size
    result["site_proxy_under_10"] = result["site_proxy_group_size"] < 10

    ordered = [
        config.EXAM_ID_COLUMN,
        "Manufacturer",
        "normalized_manufacturer",
        "ManufacturerModelName",
        "model",
        "MagneticFieldStrength",
        "SliceThickness",
        "detected_language",
        "placeholder_signature",
        "full_site_proxy_key",
        "site_proxy_key",
        "site_proxy_assignment_level",
        "site_proxy",
        "site_proxy_group_size",
        "site_proxy_under_minimum",
        "site_proxy_under_10",
    ]
    return result[[column for column in ordered if column in result.columns]]


def report_group_sizes(
    site_proxy_frame: pd.DataFrame,
    *,
    key_column: str = "site_proxy_key",
    minimum: int = 10,
) -> pd.DataFrame:
    """Return group counts and flag groups smaller than minimum."""
    if minimum < 1:
        raise ValueError("minimum must be at least 1.")
    if key_column not in site_proxy_frame.columns:
        raise ValueError(f"Missing site-proxy key column {key_column!r}.")
    sizes = (
        site_proxy_frame.groupby(key_column, dropna=False, sort=False)
        .size()
        .rename("exam_count")
        .reset_index()
    )
    sizes["under_minimum"] = sizes["exam_count"] < minimum
    sizes["under_10"] = sizes["exam_count"] < 10
    return sizes.sort_values(
        ["under_minimum", "exam_count"], ascending=[False, True], ignore_index=True
    )


def flag_small_groups(
    site_proxy_frame: pd.DataFrame,
    *,
    key_column: str = "site_proxy_key",
    minimum: int = 10,
) -> pd.DataFrame:
    """Alias for report_group_sizes with an explicit flag-oriented name."""
    return report_group_sizes(site_proxy_frame, key_column=key_column, minimum=minimum)
