import json

import numpy as np
import pandas as pd
import pytest

from src import config
from src.labeling import build_prompt, run_labeling, summarize_run
from src.soft_labels import load_soft_labels


def _reports(count=5):
    return pd.DataFrame(
        {
            config.EXAM_ID_COLUMN: [str(i) for i in range(count)],
            config.REPORT_COLUMN: ["ACL intact9xintact4cm" for _ in range(count)],
        }
    ).set_index(config.EXAM_ID_COLUMN)


def _payload(value=0.25, *, integer=False):
    values = {label: (int(value) if integer else float(value)) for label in config.TARGET_LABELS}
    return json.dumps({"probabilities": values})


def test_labeling_parses_strict_lenient_and_records_failures(tmp_path):
    prompt = tmp_path / "label_extraction_v1.md"
    prompt.write_text("Report:\n{report}\n", encoding="utf-8")
    train = tmp_path / "train.csv"
    _reports().reset_index().to_csv(train, index=False)
    fence = chr(96) * 3 + "json\n" + _payload(1.0, integer=True) + "\n" + chr(96) * 3
    trailing = _payload(0.6).replace("}}", "},}")
    responses = [_payload(0.2), fence, trailing, _payload(1.2), "x" * 600]

    def generate_fn(prompts):
        assert all("0.9x0.4cm" in prompt for prompt in prompts)
        return responses[: len(prompts)]

    results, failures = run_labeling(
        _reports(), generate_fn, prompt, tmp_path / "soft_labels.csv", batch_size=5
    )

    assert results.loc[0, "parse_status"] == "parsed"
    assert results.loc[1, "parse_status"] == "parsed_lenient"
    assert results.loc[2, "parse_status"] == "parsed_lenient"
    assert results.loc[1, "ACL"] == 1.0
    assert results.loc[2, "ACL"] == 0.6
    assert np.isnan(results.loc[4, "ACL"])
    assert len(failures) == 2
    assert len(failures.loc[failures[config.EXAM_ID_COLUMN] == "4", "raw_completion"].iloc[0]) == 500
    loaded = load_soft_labels(tmp_path / "soft_labels.csv", train_csv=train)
    assert loaded.loc["1", "ACL"] == 1.0
    assert build_prompt("x", "A {report} B") == "A x B"

    summary = summarize_run(results, failures)
    assert summary["parsed"] == 3
    assert summary["failed"] == 2
    assert summary["per_label"].loc[summary["per_label"]["label"] == "ACL", "n_at_extremes"].iloc[0] == 1


def test_labeling_resume_skips_checkpointed_ids(tmp_path):
    prompt = tmp_path / "v1.md"
    prompt.write_text("{report}", encoding="utf-8")
    output = tmp_path / "soft_labels.csv"
    reports = _reports(10)
    call_count = 0

    def interrupted(prompts):
        nonlocal call_count
        call_count += 1
        if call_count == 3:
            raise RuntimeError("simulated session loss")
        return [_payload(0.3) for _ in prompts]

    with pytest.raises(RuntimeError):
        run_labeling(reports, interrupted, prompt, output, batch_size=2, checkpoint_every=2)

    resumed_prompts = []

    def resumed(prompts):
        resumed_prompts.extend(prompts)
        return [_payload(0.7) for _ in prompts]

    results, failures = run_labeling(
        reports, resumed, prompt, output, batch_size=2, checkpoint_every=2
    )

    assert failures.empty
    assert len(results) == 10
    assert results[config.EXAM_ID_COLUMN].is_unique
    assert len(resumed_prompts) == 6
    assert results[config.EXAM_ID_COLUMN].nunique() == 10
