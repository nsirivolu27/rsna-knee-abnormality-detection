import pytest

from src.data import build_exam_index


def test_exam_index_works_in_csv_only_mode(synthetic_dataset, tmp_path):
    missing_root = tmp_path / "no-dicom-here"

    with pytest.warns(RuntimeWarning, match="CSV-only"):
        index = build_exam_index(
            train_csv=synthetic_dataset / "train.csv",
            train_series_csv=synthetic_dataset / "train_series.csv",
            series_root=missing_root,
        )

    assert len(index) == 3
    assert index["dicom_root_present"].eq(False).all()
    assert index["series_count"].tolist() == [2, 1, 1]


def test_exam_index_counts_present_series_directories(synthetic_dataset):
    index = build_exam_index(
        train_csv=synthetic_dataset / "train.csv",
        train_series_csv=synthetic_dataset / "train_series.csv",
        series_root=synthetic_dataset / "train_series",
    )

    first_exam = index.iloc[0]
    assert first_exam["dicom_root_present"]
    assert first_exam["series_directories_present"] == 2
