"""Evaluate externally generated soft labels against expert labels.

This module compares externally generated probabilities with the 58
expert-labeled exams. It does not generate labels and does not inspect report
text.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from . import config


class AgreementResult(NamedTuple):
    """Agreement table, macro-AUC, and the exam IDs used in the comparison."""

    per_label: pd.DataFrame
    macro_auc: float
    overlap_ids: pd.Index


def _coerce_frame(frame: Any, name: str) -> pd.DataFrame:
    if hasattr(frame, "values") and hasattr(frame, "observed"):
        frame = frame.values
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame or Labels object.")

    result = frame.copy()
    if config.EXAM_ID_COLUMN in result.columns:
        result = result.set_index(config.EXAM_ID_COLUMN, verify_integrity=True)
    elif result.index.name != config.EXAM_ID_COLUMN:
        raise ValueError(
            f"{name} must have {config.EXAM_ID_COLUMN!r} as a column or index name."
        )
    result.index = result.index.astype(str)
    return result


def _validate_expert(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [label for label in config.TARGET_LABELS if label not in frame.columns]
    if missing:
        raise ValueError(f"expert labels are missing columns: {missing!r}")
    values = frame.loc[:, config.TARGET_LABELS].apply(pd.to_numeric, errors="coerce")
    invalid = values.notna() & ~values.isin([0.0, 1.0])
    if invalid.any().any():
        label = next(column for column in values.columns if invalid[column].any())
        position = int(np.flatnonzero(invalid[label].to_numpy())[0])
        exam_id = values.index[position]
        raise ValueError(
            f"Invalid expert label at exam {exam_id!r}, label {label!r}: "
            f"{values.iloc[position][label]!r}. Expected 0, 1, or NaN."
        )
    return values


def _validate_derived(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [label for label in config.TARGET_LABELS if label not in frame.columns]
    if missing:
        raise ValueError(f"derived labels are missing columns: {missing!r}")
    values = frame.loc[:, config.TARGET_LABELS].apply(pd.to_numeric, errors="coerce")
    for label in config.TARGET_LABELS:
        column = values[label]
        array = column.to_numpy(dtype=float)
        invalid = column.notna().to_numpy() & (
            ~np.isfinite(array) | (array < 0.0) | (array > 1.0)
        )
        if invalid.any():
            position = int(np.flatnonzero(invalid)[0])
            exam_id = values.index[position]
            raise ValueError(
                f"Invalid derived probability at exam {exam_id!r}, label {label!r}: "
                f"{column.iloc[position]!r}. Expected a value in [0, 1] or NaN."
            )
    return values


def _auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    positives = y_true == 1
    negatives = y_true == 0
    n_positive = int(positives.sum())
    n_negative = int(negatives.sum())
    if n_positive == 0 or n_negative == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end

    positive_rank_sum = float(ranks[positives].sum())
    return (positive_rank_sum - n_positive * (n_positive + 1) / 2.0) / (
        n_positive * n_negative
    )


def _threshold_metrics(
    y_true: np.ndarray, scores: np.ndarray
) -> tuple[float, float, float, float]:
    if len(scores) == 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"))

    thresholds = np.unique(np.concatenate(([0.0, 1.0], scores)))
    candidates: list[tuple[float, float, float, float, float]] = []
    for threshold in thresholds:
        predicted = scores >= threshold
        accuracy = float(np.mean(predicted == (y_true == 1)))
        true_positive = int(np.sum(predicted & (y_true == 1)))
        true_negative = int(np.sum(~predicted & (y_true == 0)))
        n_positive = int(np.sum(y_true == 1))
        n_negative = int(np.sum(y_true == 0))
        sensitivity = true_positive / n_positive if n_positive else float("nan")
        specificity = true_negative / n_negative if n_negative else float("nan")
        candidates.append(
            (
                accuracy,
                float(threshold),
                sensitivity,
                specificity,
                abs(float(threshold) - 0.5),
            )
        )

    # Maximize accuracy; then prefer a threshold near 0.5; then the lower one.
    best = min(candidates, key=lambda item: (-item[0], item[4], item[1]))
    return best[1], best[0], best[2], best[3]


def evaluate_agreement(
    expert_labels: Any,
    derived_labels: pd.DataFrame,
) -> AgreementResult:
    """Evaluate derived probabilities against every overlapping expert exam.

    Expert NaNs are excluded per label. Derived NaNs are also excluded for
    that label. AUC is NaN when the overlap contains only one expert class;
    macro-AUC averages only defined per-label AUC values.

    derived_positive_rate is the mean derived probability, while
    expert_positive_rate is the observed binary rate. Their signed difference
    is the systematic-bias measure requested by the findings.
    """
    expert = _validate_expert(_coerce_frame(expert_labels, "expert_labels"))
    derived = _validate_derived(_coerce_frame(derived_labels, "derived_labels"))
    overlap = expert.index.intersection(derived.index)

    rows: list[dict[str, object]] = []
    for label in config.TARGET_LABELS:
        expert_column = expert.loc[overlap, label]
        derived_column = derived.loc[overlap, label]
        usable = expert_column.notna() & derived_column.notna()
        y_true = expert_column[usable].to_numpy(dtype=float)
        scores = derived_column[usable].to_numpy(dtype=float)
        n_overlap = len(y_true)
        n_positive = int(np.sum(y_true == 1))
        n_negative = int(np.sum(y_true == 0))
        if n_overlap:
            derived_rate = float(np.mean(scores))
            expert_rate = float(np.mean(y_true))
        else:
            derived_rate = float("nan")
            expert_rate = float("nan")
        threshold, accuracy, sensitivity, specificity = _threshold_metrics(y_true, scores)
        difference = derived_rate - expert_rate
        rows.append(
            {
                "label": label,
                "n_overlap": n_overlap,
                "expert_positive_count": n_positive,
                "expert_negative_count": n_negative,
                "expert_positive_rate": expert_rate,
                "derived_positive_rate": derived_rate,
                "positive_rate_difference": difference,
                "signed_positive_rate_difference": difference,
                "auc": _auc(y_true, scores),
                "best_threshold": threshold,
                "best_threshold_accuracy": accuracy,
                "sensitivity": sensitivity,
                "specificity": specificity,
            }
        )

    table = pd.DataFrame(rows)
    table = table.sort_values("auc", ascending=False, na_position="last", ignore_index=True)
    table["auc_rank"] = table["auc"].rank(
        method="min", ascending=False, na_option="bottom"
    ).astype("Int64")
    valid_auc = table["auc"].dropna()
    macro_auc = float(valid_auc.mean()) if not valid_auc.empty else float("nan")
    return AgreementResult(per_label=table, macro_auc=macro_auc, overlap_ids=overlap)
