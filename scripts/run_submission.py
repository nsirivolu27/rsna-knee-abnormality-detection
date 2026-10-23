"""Produce a submission file for the competition test set.

Runs the full path: read the test exam list, route and preprocess each exam's
images, call a predictor, assemble and validate the submission. Designed to run
inside a Kaggle notebook with internet disabled.

Until a trained model exists, the default predictor returns the per-label
positive rate observed in the expert-labeled examinations. That cannot rank
exams and so scores at chance, but it produces a well-formed submission and
exercises every stage of the pipeline.

Usage from a notebook:

    from scripts.run_submission import main
    submission, diagnostics = main()

Usage from a shell:

    python -m scripts.run_submission --output submission.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src import config
from src.inference import Predictor, constant_predictor, prior_predictor, run_inference, summarize_run
from src.labels import load_labels
from src.submission import label_priors


def default_predictor() -> Predictor:
    """Priors from the expert labels, falling back to 0.5 if unreadable."""
    try:
        priors = label_priors(load_labels())
    except Exception as error:  # noqa: BLE001 - a missing train.csv must not block a run
        print(f"Could not read expert labels ({type(error).__name__}); using 0.5.", flush=True)
        return constant_predictor(0.5)
    formatted = ", ".join(f"{label} {priors[label]:.2f}" for label in config.TARGET_LABELS)
    print(f"Label priors: {formatted}", flush=True)
    return prior_predictor(priors)


def main(
    predictor: Predictor | None = None,
    output: Path | str = "submission.csv",
    progress_every: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build, validate and write a submission. Returns (submission, diagnostics)."""
    print(f"Data root: {config.get_data_root()}", flush=True)
    submission, diagnostics = run_inference(
        predictor or default_predictor(),
        output_path=output,
        progress_every=progress_every,
    )
    print(summarize_run(diagnostics), flush=True)
    print(f"Wrote {output} with {len(submission)} rows.", flush=True)

    failures = diagnostics[diagnostics.status != "ok"]
    if len(failures):
        print(f"\n{len(failures)} exam(s) did not produce image-based predictions:", flush=True)
        print(failures.head(10).to_string(index=False), flush=True)
    return submission, diagnostics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="submission.csv", help="Where to write the CSV.")
    parser.add_argument("--progress-every", type=int, default=50, help="Progress interval, 0 to silence.")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    main(output=arguments.output, progress_every=arguments.progress_every)
