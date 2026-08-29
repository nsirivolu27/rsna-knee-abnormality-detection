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
    assert set(result.label_prevalence["fold"]) == {0, 1, 2}
    assert result.label_prevalence["positive_rate"].notna().all()
