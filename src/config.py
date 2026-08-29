"""Central configuration for the RSNA knee EDA project.

This module contains constants only. It must remain importable without reading
data, importing optional libraries, or running analysis.
"""

import os
from pathlib import Path
from typing import Final


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

# The notebook sets this before importing src.config on Kaggle. The local
# default keeps imports and future synthetic smoke tests usable without the
# competition data.
DATA_ROOT: Final[Path] = Path(
    os.environ.get("RSNA_KNEE_DATA_ROOT", PROJECT_ROOT / "data")
)

RANDOM_SEED: Final[int] = 42

# ---------------------------------------------------------------------------
# UNVERIFIED SCHEMA PLACEHOLDERS
# ---------------------------------------------------------------------------
# These names are deliberately not inferred from public material. Replace
# them only after the competition owner supplies the Data-tab file/folder
# listing, every CSV header with three sample rows, and the exact 12 labels.
SCHEMA_IS_VERIFIED: Final[bool] = False

TARGET_LABELS: Final[tuple[str, ...]] = tuple(
    f"UNVERIFIED_LABEL_{index:02d}" for index in range(1, 13)
)

UNVERIFIED_SCHEMA: Final[dict[str, object]] = {
    "data_files": (
        "UNVERIFIED_DATA_FILE_1.csv",
        "UNVERIFIED_DATA_FILE_2.csv",
    ),
    "exam_id_column": "UNVERIFIED_EXAM_ID_COLUMN",
    "patient_id_column": "UNVERIFIED_PATIENT_ID_COLUMN",
    "series_id_column": "UNVERIFIED_SERIES_ID_COLUMN",
    "report_text_column": "UNVERIFIED_REPORT_TEXT_COLUMN",
    "language_column": "UNVERIFIED_LANGUAGE_COLUMN",
    "site_column": "UNVERIFIED_SITE_COLUMN",
    "label_columns": TARGET_LABELS,
}

# Data directories are also provisional until the Data-tab file/folder listing
# is available. No loader should consume these placeholders yet.
UNVERIFIED_DATA_PATHS: Final[dict[str, Path]] = {
    "dicom": DATA_ROOT / "UNVERIFIED_DICOM_DIRECTORY",
    "tabular": DATA_ROOT / "UNVERIFIED_TABULAR_DIRECTORY",
    "reports": DATA_ROOT / "UNVERIFIED_REPORTS_DIRECTORY",
}

# DICOM attributes planned for the metadata inventory. These are standard
# DICOM concepts, not competition CSV column names.
DICOM_METADATA_FIELDS: Final[tuple[str, ...]] = (
    "SeriesDescription",
    "ImageOrientationPatient",
    "SliceThickness",
    "SpacingBetweenSlices",
    "PixelSpacing",
    "Manufacturer",
    "MagneticFieldStrength",
)
