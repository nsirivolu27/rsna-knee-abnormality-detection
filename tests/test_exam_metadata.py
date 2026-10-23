import pandas as pd

from src.dicom_io import exam_metadata


def test_exam_metadata_prefers_sagittal_and_roundtrips_parquet(synthetic_dataset, tmp_path):
    study_id = "1.2.826.0.1.3680043.8.498.100.1"
    cache = tmp_path / "metadata.parquet"
    result = exam_metadata(
        exams=[study_id],
        series_csv=synthetic_dataset / "train_series.csv",
        series_root=synthetic_dataset / "train_series",
        cache_path=cache,
    )

    assert len(result) == 1
    assert result.loc[0, "selected_series_uid"].endswith(".101")
    assert result.loc[0, "selection_strategy"] == "sagittal"
    assert result.loc[0, "Manufacturer"] == "Synthetic"
    assert pd.isna(result.loc[0, "warning_code"])
    assert cache.exists()

    cached = exam_metadata(cache_path=cache)
    pd.testing.assert_frame_equal(result, cached)
