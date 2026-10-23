"""Notebook-friendly labeling runner; model creation stays with the caller."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src import config
from src.agreement import AgreementResult, evaluate_agreement
from src.labeling import run_labeling
from src.labels import load_labels
from src.reports import load_reports
from src.soft_labels import load_soft_labels


def main(
    generate_fn: Callable[[list[str]], list[str]],
    reports: pd.DataFrame | pd.Series | Path | str | None = None,
    *,
    prompt_path: Path | str = config.PROJECT_ROOT / "prompts" / "label_extraction_v1.md",
    out_path: Path | str = "soft_labels.csv",
    expert_labels: Any = None,
    train_csv: Path | str | None = None,
    batch_size: int = 8,
    checkpoint_every: int = 50,
    resume: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, AgreementResult]:
    """Run labeling, validate the output, and print the agreement table."""
    report_frame = load_reports(reports) if isinstance(reports, (Path, str)) else reports
    if report_frame is None:
        report_frame = load_reports()
    results, failures = run_labeling(
        report_frame,
        generate_fn,
        prompt_path,
        out_path,
        batch_size=batch_size,
        checkpoint_every=checkpoint_every,
        resume=resume,
    )
    validated = load_soft_labels(out_path, train_csv=train_csv)
    expert = load_labels() if expert_labels is None else expert_labels
    agreement = evaluate_agreement(expert, validated)
    print(agreement.per_label.to_string(index=False))
    return results, failures, agreement
