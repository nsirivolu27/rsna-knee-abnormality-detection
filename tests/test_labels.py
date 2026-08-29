from src import config
from src.labels import fully_labeled_exams, load_labels, summarize_labels


def test_load_labels_preserves_unknowns_and_exam_index(synthetic_dataset):
    labels = load_labels(synthetic_dataset / "train.csv")

    assert labels.values.index.name == config.EXAM_ID_COLUMN
    assert labels.values.shape == (3, 12)
    assert labels.observed.dtypes.eq(bool).all()
    assert labels.observed.iloc[1].eq(False).all()
    assert labels.values.iloc[1].isna().all()


def test_label_summary_reports_three_explicit_counts(synthetic_dataset):
    labels = load_labels(synthetic_dataset / "train.csv")
    summary = summarize_labels(labels)

    assert list(summary.columns) == ["positive", "negative", "missing"]
    assert summary.loc["ACL"].to_dict() == {
        "positive": 1,
        "negative": 1,
        "missing": 1,
    }


def test_fully_labeled_exams_are_validation_rows(synthetic_dataset):
    labels = load_labels(synthetic_dataset / "train.csv")
    complete = fully_labeled_exams(labels)

    assert complete.index.name == config.EXAM_ID_COLUMN
    assert len(complete) == 2
    assert complete.notna().all().all()
