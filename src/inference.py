"""End-to-end prediction path: test exams in, submission frame out.

The model is injected rather than imported. This module owns the loop, the
failure handling and the assembly; it knows nothing about architectures, and
imports no deep learning framework, so it stays testable without one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import config
from .cache import VolumeCache
from .dataset import ExamDataset
from .preprocessing import PreprocessConfig
from .series_selection import SLOT_NAMES
from .submission import build_submission, validate_submission

# A predictor receives one prepared exam and returns twelve probabilities in
# config.TARGET_LABELS order. The exam dict is what ExamDataset yields: an
# images array of shape (slots, slices, height, width) and a presence_mask.
Predictor = Callable[[Mapping[str, object]], Sequence[float]]


def constant_predictor(value: float = 0.5) -> Predictor:
    """Predict the same probability for every label on every exam."""
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError("value must lie in [0, 1].")
    vector = np.full(len(config.TARGET_LABELS), float(value), dtype=float)

    def predict(_exam: Mapping[str, object]) -> Sequence[float]:
        return vector.copy()

    return predict


def prior_predictor(priors: Mapping[str, float]) -> Predictor:
    """Predict each label's prior positive rate, ignoring the images.

    This is the honest floor for this task. It cannot rank exams, so it scores
    at chance under AUC, but it produces a well-formed submission and exercises
    the full path.
    """
    missing = sorted(set(config.TARGET_LABELS) - set(priors))
    if missing:
        raise ValueError(f"priors is missing labels: {missing!r}")
    vector = np.array([float(priors[label]) for label in config.TARGET_LABELS], dtype=float)
    if vector.min() < 0.0 or vector.max() > 1.0:
        raise ValueError("Every prior must lie in [0, 1].")

    def predict(_exam: Mapping[str, object]) -> Sequence[float]:
        return vector.copy()

    return predict


def _coerce_prediction(raw: object) -> np.ndarray:
    """Turn whatever the predictor returned into twelve valid probabilities."""
    if isinstance(raw, Mapping):
        missing = sorted(set(config.TARGET_LABELS) - set(raw))
        if missing:
            raise ValueError(f"Predictor omitted labels: {missing!r}")
        vector = np.array([float(raw[label]) for label in config.TARGET_LABELS], dtype=float)
    else:
        vector = np.asarray(raw, dtype=float).reshape(-1)
        if vector.size != len(config.TARGET_LABELS):
            raise ValueError(
                f"Predictor returned {vector.size} values, expected {len(config.TARGET_LABELS)}."
            )
    if not np.isfinite(vector).all():
        raise ValueError("Predictor returned a non-finite probability.")
    if vector.min() < 0.0 or vector.max() > 1.0:
        raise ValueError(
            f"Predictor returned a value outside [0, 1]; range "
            f"[{vector.min():.4f}, {vector.max():.4f}]"
        )
    return vector


def predict_exams(
    exam_ids: Iterable[str],
    predictor: Predictor,
    series_frame: pd.DataFrame | None = None,
    series_csv: Path | str | None = None,
    series_root: Path | str | None = None,
    preprocess_config: PreprocessConfig | None = None,
    cache: VolumeCache | None = None,
    cache_dir: Path | str | None = None,
    fallback: Mapping[str, float] | float = 0.5,
    progress_every: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the pipeline over exams and return (predictions, diagnostics).

    Every requested exam gets a row. An exam whose images cannot be read, or
    whose predictor call raises, falls back rather than aborting the run: a
    crash partway through a scored inference pass costs the whole submission,
    and a fallback row costs one exam. Both cases are recorded in diagnostics.
    """
    ids = [str(value) for value in exam_ids]
    if len(ids) != len(set(ids)):
        raise ValueError("exam_ids must not contain duplicates.")

    if isinstance(fallback, Mapping):
        missing = sorted(set(config.TARGET_LABELS) - set(fallback))
        if missing:
            raise ValueError(f"fallback is missing labels: {missing!r}")
        fallback_vector = np.array(
            [float(fallback[label]) for label in config.TARGET_LABELS], dtype=float
        )
    else:
        fallback_vector = np.full(len(config.TARGET_LABELS), float(fallback), dtype=float)

    dataset = ExamDataset(
        exam_ids=ids,
        series_frame=series_frame,
        series_csv=series_csv if series_csv is not None else config.TEST_SERIES_CSV,
        series_root=series_root if series_root is not None else config.TEST_SERIES_ROOT,
        preprocess_config=preprocess_config,
        cache=cache,
        cache_dir=cache_dir,
    )

    rows: list[np.ndarray] = []
    diagnostics: list[dict[str, object]] = []
    for position in range(len(dataset)):
        exam_id = ids[position]
        status = "ok"
        detail = ""
        slots_present = 0
        try:
            item = dataset[position]
            slots_present = int(np.asarray(item["presence_mask"], dtype=bool).sum())
            vector = _coerce_prediction(predictor(item))
            if slots_present == 0:
                status = "no_images"
                detail = "No canonical slot could be loaded for this exam."
        except Exception as error:  # noqa: BLE001 - one bad exam must not end the run
            vector = fallback_vector.copy()
            status = "failed"
            detail = f"{type(error).__name__}: {error}"
        rows.append(vector)
        diagnostics.append(
            {
                config.EXAM_ID_COLUMN: exam_id,
                "status": status,
                "slots_present": slots_present,
                "detail": detail,
            }
        )
        if progress_every and (position + 1) % progress_every == 0:
            print(f"{position + 1}/{len(ids)} exams", flush=True)

    predictions = pd.DataFrame(rows, index=pd.Index(ids, name=config.EXAM_ID_COLUMN),
                               columns=list(config.TARGET_LABELS))
    return predictions, pd.DataFrame(diagnostics)


def run_inference(
    predictor: Predictor,
    exam_ids: Iterable[str] | None = None,
    test_csv: Path | str | None = None,
    output_path: Path | str = "submission.csv",
    write: bool = True,
    **kwargs: object,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict the test set and produce a validated submission frame.

    Reads the test exam list from test.csv when exam_ids is not supplied, so
    the same call works against both the three-row placeholder test set and the
    hidden test set substituted at scoring time.
    """
    if exam_ids is None:
        test_frame = pd.read_csv(test_csv if test_csv is not None else config.TEST_CSV)
        exam_ids = test_frame[config.EXAM_ID_COLUMN].astype(str).tolist()
    ids = [str(value) for value in exam_ids]

    predictions, diagnostics = predict_exams(ids, predictor, **kwargs)  # type: ignore[arg-type]
    submission = build_submission(predictions, exam_ids=ids)
    validate_submission(submission, expected_exam_ids=ids)
    if write:
        submission.to_csv(output_path, index=False)
    return submission, diagnostics


def summarize_run(diagnostics: pd.DataFrame) -> str:
    """One-line human summary of an inference pass."""
    if diagnostics.empty:
        return "no exams processed"
    counts = diagnostics.status.value_counts()
    parts = [f"{int(count)} {name}" for name, count in counts.items()]
    mean_slots = diagnostics.slots_present.mean()
    return (
        f"{len(diagnostics)} exams: " + ", ".join(parts)
        + f" | mean slots present {mean_slots:.2f} of {len(SLOT_NAMES)}"
    )
