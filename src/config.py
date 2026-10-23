"""Central configuration for the RSNA knee EDA project.

This module contains constants and small path helpers only. Importing it does
not read data, import optional libraries, or run analysis.
"""

import os
from pathlib import Path
from typing import Final


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

# Kaggle has used both of these mount layouts for attached competition data.
KAGGLE_DATA_ROOT: Final[Path] = Path(
    "/kaggle/input/competitions/rsna-knee-abnormality-detection"
)
ALTERNATE_KAGGLE_DATA_ROOT: Final[Path] = Path(
    "/kaggle/input/rsna-knee-abnormality-detection"
)
KAGGLE_DATA_ROOTS: Final[tuple[Path, ...]] = (
    KAGGLE_DATA_ROOT,
    ALTERNATE_KAGGLE_DATA_ROOT,
)

# This directory is reserved for a small, manually exported subset used only
# for local loader development and pytest fixtures. It is never the EDA data.
LOCAL_SUBSET_ROOT: Path = Path(
    os.environ.get(
        "RSNA_KNEE_LOCAL_SUBSET_ROOT",
        str(PROJECT_ROOT / "local_subset"),
    )
)


def _find_existing_kaggle_root() -> Path | None:
    """Return the first existing Kaggle mount, if either one exists."""
    for candidate in KAGGLE_DATA_ROOTS:
        if candidate.exists():
            return candidate
    return None


def _initial_data_root() -> Path:
    """Choose the configured root without requiring data at import time."""
    configured_root = os.environ.get("RSNA_KNEE_DATA_ROOT")
    if configured_root:
        return Path(configured_root)

    existing_root = _find_existing_kaggle_root()
    if existing_root is not None:
        return existing_root

    # Keep imports usable in the local authoring environment. A call to
    # get_data_root() performs the strict missing-mount check.
    return KAGGLE_DATA_ROOT


DATA_ROOT: Path = _initial_data_root()

TRAIN_CSV: Path = DATA_ROOT / "train.csv"
TRAIN_SERIES_CSV: Path = DATA_ROOT / "train_series.csv"
TEST_CSV: Path = DATA_ROOT / "test.csv"
TEST_SERIES_CSV: Path = DATA_ROOT / "test_series.csv"
SAMPLE_SUBMISSION_CSV: Path = DATA_ROOT / "sample_submission.csv"
TRAIN_SERIES_ROOT: Path = DATA_ROOT / "train_series"
TEST_SERIES_ROOT: Path = DATA_ROOT / "test_series"


def set_data_root(root: Path | str) -> None:
    """Point the module at a different data root at runtime."""
    global DATA_ROOT
    global TRAIN_CSV, TRAIN_SERIES_CSV, TEST_CSV, TEST_SERIES_CSV
    global SAMPLE_SUBMISSION_CSV, TRAIN_SERIES_ROOT, TEST_SERIES_ROOT

    DATA_ROOT = Path(root)
    TRAIN_CSV = DATA_ROOT / "train.csv"
    TRAIN_SERIES_CSV = DATA_ROOT / "train_series.csv"
    TEST_CSV = DATA_ROOT / "test.csv"
    TEST_SERIES_CSV = DATA_ROOT / "test_series.csv"
    SAMPLE_SUBMISSION_CSV = DATA_ROOT / "sample_submission.csv"
    TRAIN_SERIES_ROOT = DATA_ROOT / "train_series"
    TEST_SERIES_ROOT = DATA_ROOT / "test_series"


def get_data_root() -> Path:
    """Return the active data root, validating the Kaggle mount if needed."""
    if DATA_ROOT in KAGGLE_DATA_ROOTS and not DATA_ROOT.exists():
        existing_root = _find_existing_kaggle_root()
        if existing_root is not None:
            set_data_root(existing_root)
        else:
            checked_paths = "\n".join(f"  - {path}" for path in KAGGLE_DATA_ROOTS)
            raise FileNotFoundError(
                "Kaggle competition data was not found. Checked both paths:\n"
                f"{checked_paths}\n"
                "Attach the competition in Kaggle or call "
                "set_data_root() with a local subset path."
            )
    return DATA_ROOT


RANDOM_SEED: Final[int] = 42

EXAM_ID_COLUMN: Final[str] = "StudyInstanceUID"
REPORT_COLUMN: Final[str] = "Report"

TARGET_LABELS: Final[tuple[str, ...]] = (
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
)

SERIES_ID_COLUMN: Final[str] = "SeriesInstanceUID"
FLUID_SENSITIVE_COLUMN: Final[str] = "Fluid_Sensitive"
FAT_SUPPRESSION_COLUMN: Final[str] = "Fat_Suppression"
ANATOMICAL_PLANE_COLUMN: Final[str] = "Anatomical_Plane"

# DICOM attributes planned for the metadata inventory. These are standard
# DICOM concepts, not competition CSV column names.
DICOM_METADATA_FIELDS: Final[tuple[str, ...]] = (
    "Manufacturer",
    "ManufacturerModelName",
    "StationName",
    "MagneticFieldStrength",
    "InstitutionName",
    "PixelSpacing",
    "SliceThickness",
    "ImageOrientationPatient",
    "SeriesDescription",
    "SpacingBetweenSlices",
)
