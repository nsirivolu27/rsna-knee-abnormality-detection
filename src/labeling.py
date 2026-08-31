"""Pure, resumable report-to-soft-label orchestration.

The model runtime is deliberately injected through generate_fn. This module does
not import a model library or know how completions are produced.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from . import config
from . import reports as report_utils

GenerateFn = Callable[[list[str]], list[str]]
_METADATA_COLUMNS = (
    "prompt_sha",
    "prompt_version",
    "parse_status",
    "raw_completion",
    "parse_error",
    "rationale",
)
_FAILURE_COLUMNS = (config.EXAM_ID_COLUMN, "raw_completion", "parse_error")


def build_prompt(report_text: str, template: str) -> str:
    """Insert one report into a prompt template without interpreting JSON braces."""
    if "{report}" not in template:
        raise ValueError("Prompt template must contain the {report} placeholder.")
    return template.replace("{report}", str(report_text))


def _coerce_reports(reports: pd.DataFrame | pd.Series) -> pd.DataFrame:
    if isinstance(reports, pd.Series):
        frame = reports.to_frame(name=config.REPORT_COLUMN)
    elif isinstance(reports, pd.DataFrame):
        frame = reports.copy()
    else:
        raise TypeError("reports must be a pandas DataFrame or Series.")

    if config.EXAM_ID_COLUMN in frame.columns:
        frame = frame.set_index(config.EXAM_ID_COLUMN, verify_integrity=True)
    elif frame.index.name != config.EXAM_ID_COLUMN:
        raise ValueError(
            f"reports must have {config.EXAM_ID_COLUMN!r} as a column or index name."
        )
    if config.REPORT_COLUMN not in frame.columns:
        raise ValueError(f"reports must contain {config.REPORT_COLUMN!r}.")
    frame.index = frame.index.astype(str)
    if not frame.index.is_unique:
        raise ValueError("reports contain duplicate exam IDs.")
    return frame


def _read_output(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    if path.suffix.casefold() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _validate_existing(existing: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        return existing
    if config.EXAM_ID_COLUMN not in existing.columns:
        raise ValueError(f"Existing output must contain {config.EXAM_ID_COLUMN!r}.")
    if existing[config.EXAM_ID_COLUMN].isna().any():
        raise ValueError("Existing output contains a missing exam ID.")
    existing = existing.copy()
    existing[config.EXAM_ID_COLUMN] = existing[config.EXAM_ID_COLUMN].astype(str)
    if existing[config.EXAM_ID_COLUMN].duplicated().any():
        raise ValueError("Existing output contains duplicate exam IDs.")
    return existing


def _append_output(path: Path, rows: pd.DataFrame) -> None:
    if rows.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.casefold() == ".parquet":
        combined = _read_output(path)
        combined = pd.concat([combined, rows], ignore_index=True)
        temporary = path.with_name(path.name + ".tmp")
        combined.to_parquet(temporary, index=False)
        temporary.replace(path)
        return
    header = not path.exists() or path.stat().st_size == 0
    rows.to_csv(path, mode="a", header=header, index=False)


def _balanced_object(text: str) -> str | None:
    """Return the first balanced JSON object, respecting quoted braces."""
    for start, character in enumerate(text):
        if character != "{":
            continue
        depth = 0
        in_string = False
        escaped = False
        for position in range(start, len(text)):
            current = text[position]
            if in_string:
                if escaped:
                    escaped = False
                elif ord(current) == 92:
                    escaped = True
                elif ord(current) == 92:
                    in_string = False
                continue
            if current == '"':
                in_string = True
                elif ord(current) == 92:
                depth += 1
                elif ord(current) == 92:
                depth -= 1
                if depth == 0:
                    return text[start : position + 1]
    return None


def _clean_lenient(text: str) -> str:
    fence = chr(96) * 3
    without_fences = re.sub(re.escape(fence) + r"(?:json|JSON)?\s*", "", text)
    without_fences = without_fences.replace(fence, "")
    return re.sub(r",\s*([}\]])", r"\1", without_fences)


def _probability(value: Any, label: str, *, lenient: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label!r} is not numeric")
    if not lenient and not isinstance(value, float):
        raise ValueError(f"{label!r} must be a JSON float")
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{label!r} is outside [0, 1]")
    return result


def _parse_payload(candidate: str, *, lenient: bool) -> tuple[dict[str, float], dict[str, float], str | None]:
    payload = json.loads(candidate)
    if not isinstance(payload, dict):
        raise ValueError("completion JSON must be an object")
    probabilities = payload.get("probabilities")
    if not isinstance(probabilities, dict):
        raise ValueError("completion must contain a probabilities object")
    parsed = {
        label: _probability(probabilities[label], label, lenient=lenient)
        for label in config.TARGET_LABELS
    }
    confidence_payload = payload.get("confidence")
    confidence: dict[str, float] = {}
    if confidence_payload is not None:
        if not isinstance(confidence_payload, dict):
            raise ValueError("confidence must be an object when present")
        confidence = {
            label: _probability(confidence_payload[label], label, lenient=lenient)
            for label in config.TARGET_LABELS
        }
    rationale = payload.get("rationale")
    if rationale is not None and not isinstance(rationale, str):
        rationale = str(rationale)
    return parsed, confidence, rationale


def _parse_completion(completion: Any) -> tuple[dict[str, float], dict[str, float], str | None, str]:
    text = str(completion)
    candidate = _balanced_object(text)
    if candidate is not None:
        try:
            probabilities, confidence, rationale = _parse_payload(candidate, lenient=False)
            return probabilities, confidence, rationale, "parsed"
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as strict_error:
            strict_message = str(strict_error)
    else:
        strict_message = "no balanced JSON object found"

    cleaned = _clean_lenient(text)
    candidate = _balanced_object(cleaned)
    if candidate is None:
        raise ValueError(strict_message)
    try:
        probabilities, confidence, rationale = _parse_payload(candidate, lenient=True)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as lenient_error:
        raise ValueError(str(lenient_error)) from lenient_error
    return probabilities, confidence, rationale, "parsed_lenient"


def _result_row(
    exam_id: str,
    completion: Any,
    prompt_sha: str,
    prompt_version: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    row: dict[str, Any] = {
        config.EXAM_ID_COLUMN: exam_id,
        **{label: np.nan for label in config.TARGET_LABELS},
        **{f"confidence_{label}": np.nan for label in config.TARGET_LABELS},
        "prompt_sha": prompt_sha,
        "prompt_version": prompt_version,
        "parse_status": "failed",
        "raw_completion": str(completion)[:500],
        "parse_error": None,
        "rationale": None,
    }
    try:
        probabilities, confidence, rationale, status = _parse_completion(completion)
    except ValueError as error:
        row["parse_error"] = str(error)
        failure = {
            config.EXAM_ID_COLUMN: exam_id,
            "raw_completion": str(completion)[:500],
            "parse_error": str(error),
        }
        return row, failure

    row.update(probabilities)
    row.update({f"confidence_{label}": value for label, value in confidence.items()})
    row["parse_status"] = status
    row["raw_completion"] = None
    row["rationale"] = rationale
    return row, None


def run_labeling(
    reports: pd.DataFrame | pd.Series,
    generate_fn: GenerateFn,
    prompt_path: Path | str,
    out_path: Path | str,
    *,
    batch_size: int = 8,
    checkpoint_every: int = 50,
    resume: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate, checkpoint, and resume probabilistic report labels.

    The generator receives a batch of rendered prompts and must return one raw
    completion per prompt. Failed parses are checkpointed as rows with NaN label
    probabilities, and are also returned in the failures frame. Existing rows,
    including failed rows, are skipped when resume=True.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be at least 1.")

    report_frame = _coerce_reports(reports)
    prompt_file = Path(prompt_path)
    template_bytes = prompt_file.read_bytes()
    template = template_bytes.decode("utf-8")
    prompt_sha = hashlib.sha256(template_bytes).hexdigest()
    prompt_version = prompt_file.name
    output_file = Path(out_path)

    if output_file.exists() and not resume:
        raise FileExistsError(f"Output already exists and resume=False: {output_file}")
    existing = _validate_existing(_read_output(output_file)) if resume else pd.DataFrame()
    if not existing.empty and "prompt_sha" in existing.columns:
        existing_shas = set(existing["prompt_sha"].dropna().astype(str))
        if existing_shas and existing_shas != {prompt_sha}:
            raise ValueError("Existing output was produced by a different prompt revision.")

    completed = set(existing[config.EXAM_ID_COLUMN].astype(str)) if not existing.empty else set()
    pending = report_frame.loc[~report_frame.index.isin(completed)]
    checkpoint_rows: list[dict[str, Any]] = []
    since_checkpoint = 0

    def flush() -> None:
        nonlocal checkpoint_rows, since_checkpoint
        if checkpoint_rows:
            _append_output(output_file, pd.DataFrame(checkpoint_rows))
            checkpoint_rows = []
            since_checkpoint = 0

    pending_ids = list(pending.index)
    for start in range(0, len(pending_ids), batch_size):
        batch_ids = pending_ids[start : start + batch_size]
        prompts = [
            build_prompt(
                report_utils.normalize_report(pending.loc[exam_id, config.REPORT_COLUMN]),
                template,
            )
            for exam_id in batch_ids
        ]
        completions = list(generate_fn(prompts))
        if len(completions) != len(batch_ids):
            raise ValueError(
                f"generate_fn returned {len(completions)} completions for {len(batch_ids)} prompts."
            )
        for exam_id, completion in zip(batch_ids, completions):
            row, _ = _result_row(exam_id, completion, prompt_sha, prompt_version)
            checkpoint_rows.append(row)
            since_checkpoint += 1
            if since_checkpoint >= checkpoint_every:
                flush()
    flush()

    results = _validate_existing(_read_output(output_file))
    if results.empty and report_frame.empty:
        empty_columns = [config.EXAM_ID_COLUMN, *config.TARGET_LABELS]
        empty_columns += [f"confidence_{label}" for label in config.TARGET_LABELS]
        empty_columns += list(_METADATA_COLUMNS)
        results = pd.DataFrame(columns=empty_columns)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_file, index=False)

    if not results.empty and "parse_status" in results.columns:
        failed_results = results[results["parse_status"] == "failed"]
        failure_frame = failed_results.loc[:, list(_FAILURE_COLUMNS)].copy()
    else:
        failure_frame = pd.DataFrame(columns=list(_FAILURE_COLUMNS))
    return results, failure_frame.reset_index(drop=True)


def summarize_run(results: pd.DataFrame, failures: pd.DataFrame) -> dict[str, Any]:
    """Summarize parse outcomes and per-label probability concentration."""
    frame = results.copy()
    if config.EXAM_ID_COLUMN not in frame.columns and frame.index.name == config.EXAM_ID_COLUMN:
        frame = frame.reset_index()
    if failures is not None:
        failure_count = len(failures)
    else:
        failure_count = int(frame.get("parse_status", pd.Series(dtype=object)).eq("failed").sum())
    parsed_count = max(len(frame) - failure_count, 0)
    rows: list[dict[str, Any]] = []
    for label in config.TARGET_LABELS:
        values = pd.to_numeric(frame[label], errors="coerce")
        observed = values.dropna()
        rows.append(
            {
                "label": label,
                "mean_probability": float(observed.mean()) if not observed.empty else float("nan"),
                "n_observed": int(observed.size),
                "n_exact_0_5": int((observed == 0.5).sum()),
                "n_at_extremes": int(observed.isin([0.0, 1.0]).sum()),
            }
        )
    per_label = pd.DataFrame(rows)
    return {
        "parsed": parsed_count,
        "failed": failure_count,
        "n_parsed": parsed_count,
        "n_failed": failure_count,
        "per_label": per_label,
        "label_means": per_label.set_index("label")["mean_probability"].to_dict(),
        "probability_distribution": per_label.set_index("label")[["n_exact_0_5", "n_at_extremes"]],
    }
