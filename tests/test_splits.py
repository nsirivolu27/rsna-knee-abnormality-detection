import pandas as pd

from src import config
from src.splits import build_grouped_folds


def test_grouped_folds_do_not_split_groups_and_exclude_experts():
    ids = [str(i) for i in range(8)]
    site = pd.DataFrame(
        {
            config.EXAM_ID_COLUMN: ids,
            "site_proxy_key": [("g0",)] * 2 + [("g1",)] * 2 + [("g2",)] * 2 + [("g3",)] * 2,
        }
    )
    labels = pd.DataFrame(
        {label: [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0] for label in config.TARGET_LABELS},
        index=ids,
    ).rename_axis(config.EXAM_ID_COLUMN)

    result = build_grouped_folds(site, labels=labels, expert_labels=labels.iloc[[0]], n_splits=3, seed=9)
    assignments = result.assignments

    assert assignments.loc["0", "is_expert_labeled"]
    assert pd.isna(assignments.loc["0", "fold"])
    assert not assignments.loc["0", "training_eligible"]
    for _, group in assignments[assignments["training_eligible"]].groupby("site_proxy_key"):
        assert group["fold"].nunique() == 1
    train_prevalence = result.label_prevalence[
        result.label_prevalence["partition"] == "train"
    ]
    expert_prevalence = result.label_prevalence[
        result.label_prevalence["partition"] == "expert"
    ]
    assert set(train_prevalence["fold"]) == {0, 1, 2}
    assert expert_prevalence["n_exams"].eq(1).all()
    assert expert_prevalence["positive_rate"].notna().all()


def test_expert_flag_is_derived_from_complete_label_rows():
    ids = [str(i) for i in range(8)]
    site = pd.DataFrame(
        {
            config.EXAM_ID_COLUMN: ids,
            "site_proxy_key": [("g0",)] * 2 + [("g1",)] * 2 + [("g2",)] * 2 + [("g3",)] * 2,
        }
    )
    values = [[1.0] * len(config.TARGET_LABELS), [0.0] * len(config.TARGET_LABELS)]
    values.extend([[float("nan")] * len(config.TARGET_LABELS) for _ in range(6)])
    labels = pd.DataFrame(values, columns=config.TARGET_LABELS, index=ids).rename_axis(
        config.EXAM_ID_COLUMN
    )

    result = build_grouped_folds(site, labels=labels, n_splits=3, seed=4)

    assert int(result.assignments["is_expert_labeled"].sum()) == 2
    assert int(result.assignments["training_eligible"].sum()) == 6
    expert = result.label_prevalence[result.label_prevalence["partition"] == "expert"]
    assert expert["n_exams"].eq(2).all()
    assert expert["n_observed"].eq(2).all()


def test_default_label_assertion_is_skipped_for_a_subset(monkeypatch):
    ids = [str(i) for i in range(4)]
    values = [[1.0] * len(config.TARGET_LABELS)]
    values.extend([[float("nan")] * len(config.TARGET_LABELS) for _ in range(3)])
    labels = pd.DataFrame(values, columns=config.TARGET_LABELS, index=ids).rename_axis(
        config.EXAM_ID_COLUMN
    )
    site = pd.DataFrame(
        {
            config.EXAM_ID_COLUMN: ids[:3],
            "site_proxy_key": [("g0",), ("g1",), ("g2",)],
        }
    )
    monkeypatch.setattr("src.splits.load_labels", lambda: labels)

    result = build_grouped_folds(site, n_splits=2, seed=3)

    assert int(result.assignments["is_expert_labeled"].sum()) == 1
    assert int(result.assignments["training_eligible"].sum()) == 2
