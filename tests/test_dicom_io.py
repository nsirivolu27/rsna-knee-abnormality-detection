from src import config
from src.dicom_io import load_series, sample_series_metadata


def test_load_series_orders_slices_and_extracts_metadata(synthetic_dataset):
    study_id = "1.2.826.0.1.3680043.8.498.100.1"
    series_id = "1.2.826.0.1.3680043.8.498.100.101"
    series_path = synthetic_dataset / "train_series" / study_id / series_id

    result = load_series(series_path)

    assert result.warnings == []
    assert result.volume is not None
    assert result.volume.shape == (3, 8, 8)
    assert result.volume.dtype.name == "uint16"
    assert result.metadata["Manufacturer"] == "Synthetic Manufacturer"
    assert result.metadata["shape"] == (3, 8, 8)


def test_missing_series_returns_structured_warning(tmp_path):
    result = load_series(tmp_path / "missing-series")

    assert result.volume is None
    assert result.warnings[0]["code"] == "missing_series"
    assert "path" in result.warnings[0]


def test_metadata_sampling_reads_selected_exam_series_only(synthetic_dataset):
    frame = sample_series_metadata(
        n_exams=1,
        seed=config.RANDOM_SEED,
        series_csv=synthetic_dataset / "train_series.csv",
        series_root=synthetic_dataset / "train_series",
    )

    assert len(frame) == 2
    assert frame["Manufacturer"].notna().all()
    assert frame["shape"].isna().all()
