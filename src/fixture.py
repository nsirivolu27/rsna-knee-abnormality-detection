"""Create a tiny local dataset for pytest and loader smoke tests."""

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage

from . import config


FixturePath = Union[Path, str]


def _uid(number: int) -> str:
    return f"1.2.826.0.1.3680043.8.498.100.{number}"


def _write_dicom(
    file_path: Path,
    study_uid: str,
    series_uid: str,
    instance_number: int,
) -> None:
    pixel_array = np.full((8, 8), instance_number, dtype=np.uint16)
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = MRImageStorage
    file_meta.MediaStorageSOPInstanceUID = _uid(10000 + instance_number)
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = _uid(20000)

    dataset = FileDataset(
        str(file_path),
        {},
        file_meta=file_meta,
        preamble=b"\x00" * 128,
    )
    dataset.SOPClassUID = MRImageStorage
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.StudyInstanceUID = study_uid
    dataset.SeriesInstanceUID = series_uid
    dataset.Modality = "MR"
    dataset.SeriesDescription = "Synthetic PD"
    dataset.Manufacturer = "Synthetic"
    dataset.ManufacturerModelName = "Synth Model"
    dataset.StationName = "SynthStation"
    dataset.InstitutionName = "SynthSite"
    dataset.MagneticFieldStrength = 1.5
    dataset.PixelSpacing = [0.5, 0.5]
    dataset.SliceThickness = 3.0
    dataset.SpacingBetweenSlices = 3.0
    dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    dataset.ImagePositionPatient = [0.0, 0.0, float(instance_number)]
    dataset.InstanceNumber = instance_number
    dataset.Rows, dataset.Columns = pixel_array.shape
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.PixelData = pixel_array.tobytes()
    dataset.save_as(file_path)


def create_synthetic_dataset(root: FixturePath) -> Path:
    """Write a tiny train/series dataset and return its root directory."""
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)

    study_ids = [_uid(1), _uid(2), _uid(3)]
    series_ids = [_uid(101), _uid(102), _uid(201), _uid(301)]

    label_rows = [
        [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        [np.nan] * len(config.TARGET_LABELS),
        [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
    ]
    train_rows = []
    reports = [
        "The ACL is intact. Measurement intact9xintact4cm.",
        "Ήπια συλλογή υγρού στην άρθρωση.",
        "Leve derrame articular; sin fractura.",
    ]
    for study_id, report, label_row in zip(study_ids, reports, label_rows):
        row = {config.EXAM_ID_COLUMN: study_id, config.REPORT_COLUMN: report}
        row.update(dict(zip(config.TARGET_LABELS, label_row)))
        train_rows.append(row)
    pd.DataFrame(train_rows).to_csv(root_path / "train.csv", index=False)

    series_rows = [
        [study_ids[0], series_ids[0], 1, 1, "Sagittal"],
        [study_ids[0], series_ids[1], 1, 0, "Coronal"],
        [study_ids[1], series_ids[2], 0, 1, "Axial"],
        [study_ids[2], series_ids[3], 1, 1, "Sagittal"],
    ]
    series_columns = [
        config.EXAM_ID_COLUMN,
        config.SERIES_ID_COLUMN,
        config.FLUID_SENSITIVE_COLUMN,
        config.FAT_SUPPRESSION_COLUMN,
        config.ANATOMICAL_PLANE_COLUMN,
    ]
    pd.DataFrame(series_rows, columns=series_columns).to_csv(
        root_path / "train_series.csv", index=False
    )

    pd.DataFrame({config.EXAM_ID_COLUMN: study_ids}).to_csv(
        root_path / "test.csv", index=False
    )

    train_series_root = root_path / "train_series"
    for study_id, series_id, slice_count in [
        (study_ids[0], series_ids[0], 3),
        (study_ids[0], series_ids[1], 2),
        (study_ids[1], series_ids[2], 2),
        (study_ids[2], series_ids[3], 3),
    ]:
        series_path = train_series_root / study_id / series_id
        series_path.mkdir(parents=True, exist_ok=True)
        for instance_number in range(1, slice_count + 1):
            _write_dicom(
                series_path / f"slice-{instance_number:03d}.dcm",
                study_id,
                series_id,
                instance_number,
            )

    return root_path
