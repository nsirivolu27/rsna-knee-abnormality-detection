"""Lazy, framework-neutral exam dataset for canonical image slots."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from . import config
from .cache import VolumeCache
from .dicom_io import load_series
from .labels import Labels
from .preprocessing import PreprocessConfig, preprocess_volume
from .series_selection import SLOT_NAMES, select_series_map


class ExamDataset:
    """Load at most four selected DICOM series when an item is requested.

    This is intentionally not a torch Dataset: it returns NumPy arrays and
    keeps framework/model choices outside this scaffold.
    """

    def __init__(
        self,
        exam_ids: object,
        series_frame: pd.DataFrame | None = None,
        series_csv: Path | str | None = None,
        series_root: Path | str | None = None,
        labels: Labels | pd.DataFrame | None = None,
        preprocess_config: PreprocessConfig | None = None,
        cache: VolumeCache | None = None,
        cache_dir: Path | str | None = None,
    ) -> None:
        if cache is not None and cache_dir is not None:
            raise ValueError("Pass cache or cache_dir, not both.")
        values = [str(value) for value in list(exam_ids)]
        if len(values) != len(set(values)):
            raise ValueError("exam_ids must not contain duplicates.")
        self.exam_ids = values
        if series_frame is None:
            if series_csv is None:
                series_csv = config.TRAIN_SERIES_CSV
            series_frame = pd.read_csv(series_csv)
        self.series_frame = series_frame.copy()
        self.selections = select_series_map(self.series_frame, self.exam_ids)
        self.series_root = Path(series_root) if series_root is not None else config.TRAIN_SERIES_ROOT
        self.preprocess_config = preprocess_config or PreprocessConfig()
        self.cache = cache or (VolumeCache(cache_dir) if cache_dir is not None else None)
        self._label_values, self._label_observed = self._prepare_labels(labels)

    @staticmethod
    def _prepare_labels(labels: Labels | pd.DataFrame | None) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        if labels is None:
            return None, None
        if isinstance(labels, Labels):
            return labels.values, labels.observed
        frame = labels.copy()
        if config.EXAM_ID_COLUMN in frame.columns:
            frame = frame.set_index(config.EXAM_ID_COLUMN, verify_integrity=True)
        missing = sorted(set(config.TARGET_LABELS) - set(frame.columns))
        if missing:
            raise ValueError(f"labels is missing target columns: {missing!r}")
        values = frame.loc[:, config.TARGET_LABELS].astype(float)
        invalid = values.notna() & ~values.isin([0.0, 1.0])
        if invalid.any().any():
            raise ValueError("labels must contain only 0.0, 1.0, or NaN.")
        return values, values.notna()

    def __len__(self) -> int:
        return len(self.exam_ids)

    def __getitem__(self, index: int) -> dict[str, object]:
        exam_id = self.exam_ids[index]
        selection = self.selections[exam_id]
        cached = self.cache.load(exam_id, selection, self.preprocess_config) if self.cache else None
        warnings: list[dict[str, str]] = []
        if cached is not None:
            volumes = cached["volumes"]
            presence = cached["presence_mask"]
        else:
            height, width = self.preprocess_config.image_size
            shape = (self.preprocess_config.num_slices, height, width)
            volumes = {slot: np.zeros(shape, dtype=np.float32) for slot in SLOT_NAMES}
            presence = np.zeros(len(SLOT_NAMES), dtype=bool)
            for position, slot in enumerate(SLOT_NAMES):
                series_id = selection[slot]
                if series_id is None:
                    continue
                result = load_series(self.series_root / exam_id / series_id)
                warnings.extend({"slot": slot, **warning} for warning in result.warnings)
                if result.volume is None:
                    continue
                try:
                    volumes[slot] = preprocess_volume(result.volume, self.preprocess_config)
                    presence[position] = True
                except (TypeError, ValueError) as error:
                    warnings.append({"slot": slot, "code": "preprocessing_error", "message": str(error)})
            if self.cache:
                self.cache.save(exam_id, selection, self.preprocess_config, volumes, presence)

        item: dict[str, object] = {
            "exam_id": exam_id,
            "images": np.stack([volumes[slot] for slot in SLOT_NAMES], axis=0),
            "presence_mask": np.asarray(presence, dtype=bool),
            "series_uids": dict(selection),
            "warnings": warnings,
        }
        if self._label_values is not None and self._label_observed is not None:
            if exam_id in self._label_values.index:
                item["targets"] = self._label_values.loc[exam_id].to_numpy(dtype=np.float32)
                item["target_observed_mask"] = self._label_observed.loc[exam_id].to_numpy(dtype=bool)
            else:
                item["targets"] = np.full(len(config.TARGET_LABELS), np.nan, dtype=np.float32)
                item["target_observed_mask"] = np.zeros(len(config.TARGET_LABELS), dtype=bool)
        return item
