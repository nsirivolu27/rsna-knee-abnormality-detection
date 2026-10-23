"""Leak-free grouped cross-validation assignments."""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from . import config
from .labels import Labels, fully_labeled_exams, load_labels


class SplitResult(NamedTuple):
    """Exam assignments and label prevalence for training and expert partitions."""

    assignments: pd.DataFrame
    label_prevalence: pd.DataFrame


def _coerce_exam_frame(frame: Any, name: str) -> pd.DataFrame:
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


def _coerce_labels(labels: Any) -> Labels:
    if isinstance(labels, Labels):
        return labels
    frame = _coerce_exam_frame(labels, "labels")
    missing = [label for label in config.TARGET_LABELS if label not in frame.columns]
    if missing:
        raise ValueError(f"labels are missing columns: {missing!r}")
    values = frame.loc[:, config.TARGET_LABELS].apply(pd.to_numeric, errors="coerce")
    invalid = values.notna() & ~values.isin([0.0, 1.0])
    if invalid.any().any():
        label = next(column for column in values.columns if invalid[column].any())
        exam_id = values.index[invalid[label].to_numpy().nonzero()[0][0]]
        raise ValueError(f"Invalid label {label!r} at exam {exam_id!r}.")
    return Labels(values=values, observed=values.notna())


def _expert_ids(expert_labels: Any, labels: Labels) -> set[str]:
    if expert_labels is not None:
        return set(_coerce_exam_frame(expert_labels, "expert_labels").index)
    return set(fully_labeled_exams(labels).index)


def _group_token(value: object) -> str:
    if value is None or value is pd.NA:
        return "<missing>"
    return repr(value)


def _prevalence(
    assignments: pd.DataFrame,
    labels: Labels,
    n_splits: int,
) -> pd.DataFrame:
    """Report rates using only non-NaN observations, including experts separately."""
    columns = [
        "fold",
        "partition",
        "label",
        "n_exams",
        "n_observed",
        "n_positive",
        "n_negative",
        "positive_rate",
    ]
    rows: list[dict[str, object]] = []

    def add_partition(ids: pd.Index, fold: object, partition: str) -> None:
        values = labels.values.reindex(ids)
        for label in config.TARGET_LABELS:
            column = values[label]
            observed = column.notna()
            n_observed = int(observed.sum())
            n_positive = int((column[observed] == 1.0).sum())
            n_negative = int((column[observed] == 0.0).sum())
            rows.append(
                {
                    "fold": fold,
                    "partition": partition,
                    "label": label,
                    "n_exams": len(ids),
                    "n_observed": n_observed,
                    "n_positive": n_positive,
                    "n_negative": n_negative,
                    "positive_rate": (
                        n_positive / n_observed if n_observed else float("nan")
                    ),
                }
            )

    for fold in range(n_splits):
        ids = assignments.index[
            (assignments["fold"] == fold) & assignments["training_eligible"]
        ]
        add_partition(ids, fold, "train")

    expert_ids = assignments.index[assignments["is_expert_labeled"]]
    if len(expert_ids):
        add_partition(expert_ids, pd.NA, "expert")
    return pd.DataFrame(rows, columns=columns)


def build_grouped_folds(
    site_proxy_frame: pd.DataFrame,
    labels: Any = None,
    expert_labels: Any = None,
    *,
    n_splits: int = 5,
    seed: int = config.RANDOM_SEED,
    group_column: str = "site_proxy_key",
    exclude_expert: bool = True,
) -> SplitResult:
    """Assign whole site-proxy groups to seeded training folds.

    When labels is omitted, the function loads config.TRAIN_CSV and derives
    is_expert_labeled from labels.fully_labeled_exams(). On the full corpus an
    assertion requires exactly 58 complete exams. Those exams remain in the
    assignments frame with fold=<NA>, training_eligible=False, and partition
    reporting keeps their observed label prevalence separate from train folds.
    Thus default training fold sizes sum to 4,349, not 4,407.

    The resulting site-proxy key is treated as atomic and never split across
    folds. Since the observed labels are concentrated in the excluded expert
    partition, training-fold prevalence can have zero observed labels; the
    separate expert partition preserves the measured rates without treating
    NaN as negative.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if group_column not in site_proxy_frame.columns:
        raise ValueError(f"Missing grouping column {group_column!r}.")

    loaded_default_labels = labels is None
    label_source = load_labels() if loaded_default_labels else labels
    label_object = _coerce_labels(label_source)
    expert_id_set = _expert_ids(expert_labels, label_object)

    frame = _coerce_exam_frame(site_proxy_frame, "site_proxy_frame")
    result = frame.copy()
    full_corpus = (
        loaded_default_labels
        and len(result) == len(label_object.values)
        and set(result.index) == set(label_object.values.index)
    )
    if full_corpus:
        assert len(expert_id_set) == 58, (
            "The full train.csv contract expects exactly 58 fully labeled exams; "
            f"found {len(expert_id_set)}."
        )
    result["is_expert_labeled"] = result.index.to_series().isin(expert_id_set).to_numpy()
    if full_corpus:
        n_expert = int(result["is_expert_labeled"].sum())
        assert n_expert == 58, (
            "The full site-proxy frame must contain exactly 58 expert exams; "
            f"found {n_expert}."
        )
    result["training_eligible"] = ~result["is_expert_labeled"] if exclude_expert else True
    result["fold"] = pd.Series(pd.NA, index=result.index, dtype="Int64")

    eligible = result[result["training_eligible"]]
    if not eligible.empty:
        group_tokens = eligible[group_column].map(_group_token)
        group_sizes = group_tokens.value_counts(sort=False)
        if len(group_sizes) < n_splits:
            raise ValueError(
                f"Cannot make {n_splits} grouped folds from only "
                f"{len(group_sizes)} eligible groups."
            )

        rng = np.random.default_rng(seed)
        group_keys = list(group_sizes.index)
        rng.shuffle(group_keys)
        group_keys.sort(key=lambda key: int(group_sizes[key]), reverse=True)
        fold_sizes = [0] * n_splits
        group_to_fold: dict[str, int] = {}
        for key in group_keys:
            fold = min(range(n_splits), key=lambda index: (fold_sizes[index], index))
            group_to_fold[key] = fold
            fold_sizes[fold] += int(group_sizes[key])
        result.loc[eligible.index, "fold"] = group_tokens.map(group_to_fold).astype("Int64")

    prevalence = _prevalence(result, label_object, n_splits)
    return SplitResult(assignments=result, label_prevalence=prevalence)


def grouped_kfold(
    site_proxy_frame: pd.DataFrame,
    labels: Any = None,
    expert_labels: Any = None,
    **kwargs: Any,
) -> SplitResult:
    """Compatibility alias for build_grouped_folds."""
    return build_grouped_folds(
        site_proxy_frame, labels=labels, expert_labels=expert_labels, **kwargs
    )
