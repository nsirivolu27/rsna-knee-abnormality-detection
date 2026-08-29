import numpy as np
import pandas as pd

from src import config
from src.agreement import RANKING_NOTE, evaluate_agreement


def _labels(values):
    return pd.DataFrame(
        {label: values for label in config.TARGET_LABELS},
        index=["a", "b", "c", "d"],
    ).rename_axis(config.EXAM_ID_COLUMN)


def test_agreement_has_seeded_auc_intervals_and_counts():
    expert = _labels([0.0, 0.0, 1.0, 1.0])
    derived = _labels([0.1, 0.2, 0.8, 0.9])
    result = evaluate_agreement(expert, derived, n_bootstrap=100, seed=17)
    table = result.per_label

    assert {"n_positive", "n_negative", "auc_ci_lower", "auc_ci_upper"}.issubset(table.columns)
    assert table["n_positive"].eq(2).all()
    assert table["n_negative"].eq(2).all()
    assert table["auc_ci_lower"].notna().all()
    assert table["auc_ci_upper"].notna().all()
    assert table.attrs["note"] == RANKING_NOTE
    assert {"optimistic_accuracy", "optimistic_sensitivity", "optimistic_specificity"}.issubset(table.columns)
    assert "best_threshold_accuracy" not in table.columns


def test_agreement_rejects_invalid_probability_with_exam_and_label():
    expert = _labels([0.0, 0.0, 1.0, 1.0])
    derived = _labels([0.1, 0.2, 0.8, 0.9])
    derived.loc["c", "ACL"] = 1.2

    try:
        evaluate_agreement(expert, derived)
    except ValueError as error:
        message = str(error)
    else:
        raise AssertionError("expected invalid probability to be rejected")
    assert "c" in message and "ACL" in message
