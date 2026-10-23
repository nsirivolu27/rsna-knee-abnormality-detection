import pandas as pd
import pytest

from src import config
from src.soft_labels import load_soft_labels


def test_soft_labels_validate_ids_probabilities_and_optional_fields(synthetic_dataset, tmp_path):
    train = pd.read_csv(synthetic_dataset / "train.csv")
    row = {config.EXAM_ID_COLUMN: train.iloc[0][config.EXAM_ID_COLUMN]}
    row.update({label: 0.25 for label in config.TARGET_LABELS})
    row["ACL_confidence"] = 0.8
    row["rationale"] = "External review evidence."
    path = tmp_path / "soft.csv"
    pd.DataFrame([row]).to_csv(path, index=False)

    result = load_soft_labels(path, train_csv=synthetic_dataset / "train.csv")

    assert result.index.name == config.EXAM_ID_COLUMN
    assert result.loc[row[config.EXAM_ID_COLUMN], "ACL"] == 0.25
    assert result.loc[row[config.EXAM_ID_COLUMN], "rationale"] == row["rationale"]


def test_soft_labels_identify_offending_unknown_row(synthetic_dataset, tmp_path):
    row = {config.EXAM_ID_COLUMN: "not-in-train"}
    row.update({label: 0.25 for label in config.TARGET_LABELS})
    path = tmp_path / "soft.csv"
    pd.DataFrame([row]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="not-in-train"):
        load_soft_labels(path, train_csv=synthetic_dataset / "train.csv")
