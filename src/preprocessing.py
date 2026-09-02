"""Model-agnostic MRI volume preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PreprocessConfig:
    """Fixed-shape preprocessing settings for one selected series."""

    image_size: tuple[int, int] = (224, 224)
    num_slices: int = 16
    lower_percentile: float = 1.0
    upper_percentile: float = 99.0

    def __post_init__(self) -> None:
        height, width = self.image_size
        if height < 1 or width < 1:
            raise ValueError("image_size must contain positive dimensions.")
        if self.num_slices < 1:
            raise ValueError("num_slices must be positive.")
        if not 0.0 <= self.lower_percentile < self.upper_percentile <= 100.0:
            raise ValueError("Percentiles must satisfy 0 <= lower < upper <= 100.")


def normalize_volume(
    volume: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> np.ndarray:
    """Robustly scale a volume to float32 [0, 1] using only its pixels."""
    array = np.asarray(volume)
    if array.ndim != 3:
        raise ValueError(f"volume must be 3-D (slices, height, width), got {array.shape}.")
    if array.size == 0:
        raise ValueError("volume must not be empty.")
    if not 0.0 <= lower_percentile < upper_percentile <= 100.0:
        raise ValueError("Percentiles must satisfy 0 <= lower < upper <= 100.")
    values = array.astype(np.float32, copy=False)
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("volume contains no finite pixels.")
    finite_values = values[finite]
    low, high = np.percentile(finite_values, [lower_percentile, upper_percentile])
    if high <= low:
        return np.zeros(values.shape, dtype=np.float32)
    result = (values - np.float32(low)) / np.float32(high - low)
    result = np.clip(result, 0.0, 1.0)
    result[~finite] = 0.0
    return result.astype(np.float32, copy=False)


def select_slice_indices(depth: int, num_slices: int) -> np.ndarray:
    """Return deterministic nearest-neighbour depth indices spanning a volume."""
    if depth < 1 or num_slices < 1:
        raise ValueError("depth and num_slices must be positive.")
    return np.rint(np.linspace(0, depth - 1, num_slices)).astype(np.int64)


def resize_slice(image: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """Resize one 2-D image with dependency-free bilinear interpolation."""
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"image must be 2-D, got {array.shape}.")
    out_height, out_width = image_size
    if out_height < 1 or out_width < 1:
        raise ValueError("image_size must contain positive dimensions.")
    if array.shape == image_size:
        return array.copy()
    source_height, source_width = array.shape
    y = np.linspace(0.0, source_height - 1, out_height)
    x = np.linspace(0.0, source_width - 1, out_width)
    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    y1 = np.minimum(y0 + 1, source_height - 1)
    x1 = np.minimum(x0 + 1, source_width - 1)
    wy = (y - y0).astype(np.float32)[:, None]
    wx = (x - x0).astype(np.float32)[None, :]
    top = array[y0[:, None], x0[None, :]] * (1.0 - wx) + array[y0[:, None], x1[None, :]] * wx
    bottom = array[y1[:, None], x0[None, :]] * (1.0 - wx) + array[y1[:, None], x1[None, :]] * wx
    return (top * (1.0 - wy) + bottom * wy).astype(np.float32)


def preprocess_volume(volume: np.ndarray, config: PreprocessConfig | None = None) -> np.ndarray:
    """Normalize, sample depth, and resize to (num_slices, height, width)."""
    settings = config or PreprocessConfig()
    normalized = normalize_volume(
        volume,
        lower_percentile=settings.lower_percentile,
        upper_percentile=settings.upper_percentile,
    )
    indices = select_slice_indices(normalized.shape[0], settings.num_slices)
    sampled = normalized[indices]
    return np.stack(
        [resize_slice(image, settings.image_size) for image in sampled],
        axis=0,
    ).astype(np.float32, copy=False)
