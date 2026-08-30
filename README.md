# RSNA Knee Abnormality Detection

Scaffold and exploratory data analysis for the Kaggle RSNA Knee Abnormality Detection competition. This repository is intentionally limited to data inspection, DICOM/report loading utilities, site-aware validation, external soft-label validation, plotting helpers, and reproducible EDA. It does not contain model architectures, training loops, benchmarks, inference, or submission-writing code.

## Current status

The verified schema, lazy DICOM utilities, report normalization, synthetic fixture, grouped site-proxy utilities, external soft-label loader, agreement metrics, and pytest coverage are present. The EDA notebook remains for a later chunk.

The authoritative, measured record of the dataset schema, label coverage, text constraints, site proxies, and EDA implications is [docs/EDA_FINDINGS.md](docs/EDA_FINDINGS.md).

The competition data is approximately 247 GB decompressed and will not be stored on a local computer or in this repository. The full dataset is mounted read-only inside a Kaggle notebook after the competition is attached.

## Two-stage pipeline

The competition has two distinct stages:

1. **Offline report supervision:** reports from train.csv are sent to an external LLM workflow using prompts/label_extraction_v1.md. That workflow produces probability-valued soft labels and rationales outside this public repository. The prompt explicitly treats labels as image-derived and reports as potentially contradictory.
2. **Image-only downstream work:** DICOM images are paired with those externally generated labels. Reports are not available in test.csv and must not be consumed by any inference path.

There is deliberately **no rule-based text extractor** in this repository. No keyword or negation extraction logic should be added. The 58 fully labeled exams are an evaluation set for externally generated soft labels, not a training set.

## Layout

src/config.py is the single source of truth for paths, constants, and the verified schema.

~~~text
.
├── .gitignore
├── README.md
├── requirements.txt
├── prompts/
│   └── label_extraction_v1.md
├── src/
│   ├── agreement.py
│   ├── config.py
│   ├── data.py
│   ├── dicom_io.py
│   ├── fixture.py
│   ├── labels.py
│   ├── reports.py
│   ├── site_proxy.py
│   ├── soft_labels.py
│   └── splits.py
├── notebooks/
│   ├── 01_eda.ipynb
│   └── eda_findings.md
└── tests/
    ├── conftest.py
    ├── test_agreement.py
    ├── test_data.py
    ├── test_dicom_io.py
    ├── test_labels.py
    ├── test_reports.py
    ├── test_site_proxy.py
    ├── test_soft_labels.py
    └── test_splits.py
~~~

## Environment and dependencies

The dependency list uses minimum-version constraints rather than exact pins because Kaggle notebooks run a fixed image and hard pins can create conflicts. The standard Kaggle image is expected to provide NumPy, pandas, Matplotlib, seaborn, pydicom, and a parquet engine. pytest and nbstripout are local/Replit authoring tools used for tests and notebook hygiene.

The EDA notebook is authored in this repository but is run **only as a Kaggle notebook** against the mounted competition data. It is not intended to run locally against the full dataset.

For local smoke tests:

~~~bash
pip install -r requirements.txt
pytest
~~~

## Notebook hygiene

The committed EDA notebook must have all cell outputs cleared. Before committing notebook changes, run:

~~~bash
nbstripout notebooks/01_eda.ipynb
~~~

The written EDA takeaways will also be maintained in notebooks/eda_findings.md so conclusions are reviewable in a normal text diff.

## Using the package on Kaggle

The cleanest workflow is to publish this repository as a Kaggle Dataset containing the source tree, then attach that dataset and the competition dataset to the EDA notebook. The competition data remains read-only and is never copied into this repository.

In the first notebook cell, replace the repository Dataset slug with the actual slug and set the data root before importing project modules:

~~~python
import os
import sys
from pathlib import Path

repo_root = Path("/kaggle/input/<uploaded-repository-dataset>")
sys.path.insert(0, str(repo_root))
os.environ["RSNA_KNEE_DATA_ROOT"] = "/kaggle/input/competitions/rsna-knee-abnormality-detection"

from src import config
~~~

src.config.DATA_ROOT auto-detects the two supported Kaggle mount layouts and prefers the competitions/ form. The environment variable is available for the local subset override described below. After import, modules should read paths and constants from src.config; notebook cells should not scatter /kaggle/input/... paths throughout the analysis.

## Site-aware validation

src.dicom_io.exam_metadata() reads one header from a preferred sagittal series per exam (falling back to the first available series), producing full-corpus metadata with structured warning columns. Pass a parquet cache_path so the approximately 4,407 header reads happen once and the result can be reused as a Kaggle Dataset.

src.site_proxy.build_site_proxy() creates the findings-defined site-proxy key from normalized manufacturer, model, rounded field strength, detected language, and report placeholder signature. Raw manufacturer spelling is preserved. The default minimum group size is 10. Roughly 84% of exams coarsen to manufacturer-plus-language, so this is not scanner-granularity grouping. Smaller groups are visibly coarsened first to manufacturer-plus-language and then to language-only. src.splits.build_grouped_folds() treats the resulting key as atomic, so no group can span folds.

By default, build_grouped_folds() loads train.csv and derives the 58 expert-labeled exams from the complete label rows. They remain in the assignment table for auditability but have fold=<NA> and training_eligible=False, so only 4,349 exams receive training folds. Their observed-label prevalence is reported in a separate expert partition; NaN remains unknown rather than negative.

## External soft labels and agreement

External labels must contain one row per StudyInstanceUID, all twelve probability columns in [0, 1], and may include per-label confidence columns plus a rationale. src.soft_labels.load_soft_labels() validates IDs against train.csv, rejects duplicates and malformed rows, and does not commit or generate any label file.

src.agreement.evaluate_agreement() compares those probabilities with the expert set on overlapping IDs. It reports per-label AUC with a seeded 1,000-resample 95% bootstrap interval, optimistic same-sample threshold metrics explicitly labeled as upper bounds, positive-rate bias, and macro-AUC. The AUC ranking notes that 9–35 expert positives per label make nearby differences difficult to distinguish.

## Generating a small local subset

A local subset is for developing and smoke-testing loaders only, not for running the EDA conclusions. To make one without downloading the full competition:

1. Attach the competition to a Kaggle notebook.
2. Select approximately 20–50 exams after the verified exam identifier and directory layout are known.
3. Copy only those exams, including their needed DICOM files and small tabular context, from the read-only competition mount into a directory under /kaggle/working.
4. Compress that working directory and download the resulting archive.
5. Extract it locally under the configured LOCAL_SUBSET_ROOT directory.

Do not run an unrestricted recursive copy or a competition-wide download.

For local loader development, set the override before importing src.config:

~~~python
import os
from pathlib import Path

os.environ["RSNA_KNEE_DATA_ROOT"] = str(Path("local_subset"))
~~~

## DICOM handling policy

Any code that touches DICOM pixels must be lazy and per-exam. It must not load the complete dataset, glob the entire DICOM tree, or build an index that requires reading every DICOM file. The synthetic fixture exists only to make pytest runnable in the local authoring environment; it is not an EDA data substitute.

## Data safety

Do not commit DICOM files, report text, generated labels, competition downloads, model weights, caches, or credentials. The 58-report attachment and any externally generated label file are working material only.
