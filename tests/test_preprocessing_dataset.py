import numpy as np
import pandas as pd

from src import config
from src.cache import VolumeCache
from src.dataset import ExamDataset
from src.preprocessing import (
    PreprocessConfig,
    normalize_volume,
    preprocess_volume,
    select_slice_indices,
)


def test_normalize_volume_is_robust_and_bounded():
    volume = np.array([[[0, 1], [2, 1000]], [[3, 4], [5, 6]]], dtype=np.uint16)
    normalized = normalize_volume(volume, lower_percentile=0, upper_percentile=100)
    assert normalized.dtype == np.float32
    assert normalized.min() == 0.0
    assert normalized.max() == 1.0
    assert np.all((normalized >= 0) & (normalized <= 1))


def test_constant_volume_becomes_zero_and_slice_sampling_is_deterministic():
    assert np.array_equal(select_slice_indices(3, 5), np.array([0, 0, 1, 2, 2]))
    result = preprocess_volume(np.ones((2, 3, 4), dtype=np.float32), PreprocessConfig(image_size=(5, 6), num_slices=3))
    assert result.shape == (3, 5, 6)
    assert result.dtype == np.float32
    assert np.all(result == 0)


def test_cache_round_trip_and_configuration_key(tmp_path):
    settings = PreprocessConfig(image_size=(4, 4), num_slices=2)
    selection = {slot: None for slot in ("sag_fluid", "cor_fluid", "ax_fluid", "sag_t1")}
    volumes = {slot: np.full((2, 4, 4), index, dtype=np.float32) for index, slot in enumerate(selection)}
    presence = np.array([True, False, True, False])
    cache = VolumeCache(tmp_path / "cache")
    path = cache.save("exam", selection, settings, volumes, presence)
    loaded = cache.load("exam", selection, settings)
    assert path.exists()
    assert loaded is not None
    assert np.array_equal(loaded["presence_mask"], presence)
    assert np.array_equal(loaded["volumes"]["ax_fluid"], volumes["ax_fluid"])
    assert cache.load("exam", selection, PreprocessConfig(image_size=(5, 4), num_slices=2)) is None


def test_exam_dataset_is_lazy_shaped_and_preserves_label_mask(synthetic_dataset, tmp_path):
    labels = pd.read_csv(synthetic_dataset / "train.csv")
    settings = PreprocessConfig(image_size=(4, 4), num_slices=2)
    dataset = ExamDataset(
        labels[config.EXAM_ID_COLUMN].tolist(),
        series_csv=synthetic_dataset / "train_series.csv",
        series_root=synthetic_dataset / "train_series",
        labels=labels,
        preprocess_config=settings,
        cache_dir=tmp_path / "cache",
    )
    first = dataset[0]
    second = dataset[1]
    assert first["images"].shape == (4, 2, 4, 4)
    assert first["presence_mask"].tolist() == [True, True, False, False]
    assert first["target_observed_mask"].all()
    assert not second["target_observed_mask"].any()
    assert np.isnan(second["targets"]).all()
    assert second["presence_mask"].tolist() == [False, False, False, False]
    assert len(list((tmp_path / "cache").glob("*.npz"))) == 2
