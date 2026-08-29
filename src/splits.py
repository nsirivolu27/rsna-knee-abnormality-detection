"""Leak-free grouped cross-validation assignments."""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from . import config


class SplitResult(NamedTuple):
    """Exam assignments and label prevalence calculated on eligible folds."""

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


def _expert_ids(expert_labels: Any, frame: pd.DataFrame) -> set[str]:
    if expert_labels is not None:
        return set(_coerce_exam_frame(expert_labels, "expert_labels").index)
    if "is_expert_labeled" in frame.columns:
        return set(frame.index[frame["is_expert_labeled"].fillna(False).astype(bool)])
    if all(label in frame.columns for label in config.TARGET_LABELS):
        complete = frame.loc[:, config.TARGET_LABELS].notna().all(axis=1)
        return set(frame.index[complete])
    return set()


def _label_frame(labels: Any, frame: pd.DataFrame) -> pd.DataFrame | None:
    if labels is None:
        if all(label in frame.columns for label in config.TARGET_LABELS):
            return frame.loc[:, config.TARGET_LABELS]
        return None
    result = _coerce_exam_frame(labels, "labels")
    missing = [label for label in config.TARGET_LABELS if label not in result.columns]
    if missing:
        raise ValueError(f"labels are missing columns: {missing!r}")
    return result.loc[:, config.TARGET_LABELS]


def _group_token(value: object) -> str:
    if value is None or value is pd.NA:
        return "<missing>"
    return repr(value)


def _prevalence(
    assignments: pd.DataFrame,
    labels: pd.DataFrame | None,
    n_splits: int,
) -> pd.DataFrame:
    columns = [
        "fold",
        "label",
        "n_exams",
        "n_observed",
        "n_positive",
        "n_negative",
        "positive_rate",
    ]
    if labels is None:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for fold in range(n_splits):
        ids = assignments.index[
            (assignments["fold"] == fold) & assignments["training_eligible"]
        ]
        values = labels.reindex(ids)
        for label in config.TARGET_LABELS:
            column = values[label]
            observed = column.notna()
            n_observed = int(observed.sum())
            n_positive = int((column[observed] == 1.0).sum())
            n_negative = int((column[observed] == 0.0).sum())
            rows.append(
                {
                    "fold": fold,
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
    """Assign whole site-proxy groups to seeded folds.

    The 58 fully labeled expert exams are marked in is_expert_labeled and,
    by default, receive no training fold and are marked training_eligible=False.
    The fallback behavior for small site-proxy groups is performed upstream by
    src.site_proxy.build_site_proxy; this function treats the resulting key as
    atomic and never splits one key across folds.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if group_column not in site_proxy_frame.columns:
        raise ValueError(f"Missing grouping column {group_column!r}.")

    frame = _coerce_exam_frame(site_proxy_frame, "site_proxy_frame")
    expert_id_set = _expert_ids(expert_labels, frame)
    result = frame.copy()
    result["is_expert_labeled"] = result.index.to_series().isin(expert_id_set).to_numpy()
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

    prevalence = _prevalence(result, _label_frame(labels, frame), n_splits)
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
