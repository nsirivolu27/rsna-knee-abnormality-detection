"""Small, versioned on-disk cache for preprocessed exam tensors."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from .preprocessing import PreprocessConfig
from .series_selection import SLOT_NAMES


@dataclass(frozen=True)
class VolumeCache:
    """Cache one exam at a time; filenames do not expose exam identifiers."""

    root: Path | str
    version: str = "preprocess-v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))

    def _path(self, exam_id: str, selection: Mapping[str, str | None], settings: PreprocessConfig) -> Path:
        payload = {
            "version": self.version,
            "exam_id": str(exam_id),
            "selection": {slot: selection.get(slot) for slot in SLOT_NAMES},
            "settings": {
                "image_size": list(settings.image_size),
                "num_slices": settings.num_slices,
                "lower_percentile": settings.lower_percentile,
                "upper_percentile": settings.upper_percentile,
            },
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return self.root / f"{digest}.npz"

    def load(self, exam_id: str, selection: Mapping[str, str | None], settings: PreprocessConfig) -> dict[str, object] | None:
        path = self._path(exam_id, selection, settings)
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as archive:
                presence = archive["presence_mask"].astype(bool, copy=True)
                volumes = {slot: archive[f"volume_{index}"].copy() for index, slot in enumerate(SLOT_NAMES)}
            if presence.shape != (len(SLOT_NAMES),):
                return None
            return {"volumes": volumes, "presence_mask": presence}
        except (OSError, KeyError, ValueError):
            return None

    def save(
        self,
        exam_id: str,
        selection: Mapping[str, str | None],
        settings: PreprocessConfig,
        volumes: Mapping[str, np.ndarray],
        presence_mask: np.ndarray,
    ) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        target = self._path(exam_id, selection, settings)
        payload = {f"volume_{index}": np.asarray(volumes[slot], dtype=np.float32) for index, slot in enumerate(SLOT_NAMES)}
        payload["presence_mask"] = np.asarray(presence_mask, dtype=bool)
        fd, temporary_name = tempfile.mkstemp(dir=self.root, suffix=".tmp")
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, **payload)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return target
