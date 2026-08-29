# RSNA Knee Abnormality Detection

Scaffold and exploratory data analysis for the Kaggle RSNA Knee Abnormality Detection competition. This repository is intentionally limited to data inspection, DICOM/report loading utilities, label parsing, plotting helpers, and reproducible EDA. It does not contain model architectures, training loops, benchmarks, or submission-writing code.

## Current status

The verified schema is implemented in the central config. The data loaders, DICOM utilities, report normalization, synthetic fixture, and pytest coverage are now present; the EDA notebook remains for a later chunk.

The competition data is approximately 247 GB decompressed and will not be stored on a local computer or in this repository. The full dataset is mounted read-only inside a Kaggle notebook after the competition is attached.

## Planned layout

`src/config.py` is the single source of truth for paths, constants, and the verified schema. The EDA notebook and written findings remain to be added in later scoped chunks.

```text
.
├── .gitignore
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── dicom_io.py
│   ├── fixture.py
│   ├── labels.py
│   ├── plotting.py
│   └── reports.py
├── notebooks/
│   ├── 01_eda.ipynb
│   └── eda_findings.md
└── tests/
    ├── conftest.py
    ├── test_data.py
    ├── test_dicom_io.py
    ├── test_labels.py
    └── test_reports.py
```

## Environment and dependencies

The dependency list uses minimum-version constraints rather than exact pins because Kaggle notebooks run a fixed image and hard pins can create conflicts. The standard Kaggle image is expected to provide NumPy, pandas, Matplotlib, seaborn, and pydicom. pytest and nbstripout are local/Replit authoring tools used for tests and notebook hygiene.

The EDA notebook is authored in this repository but is run **only as a Kaggle notebook** against the mounted competition data. It is not intended to run locally against the full dataset.

For local smoke tests after the test files are added:

```bash
pip install -r requirements.txt
pytest
```

## Notebook hygiene

The committed EDA notebook must have all cell outputs cleared. Before committing notebook changes, run:

```bash
nbstripout notebooks/01_eda.ipynb
```

The written EDA takeaways will also be maintained in `notebooks/eda_findings.md` so conclusions are reviewable in a normal text diff.

## Using the package on Kaggle

The cleanest workflow is to publish this repository as a Kaggle Dataset containing the source tree, then attach that dataset and the competition dataset to the EDA notebook. The competition data remains read-only and is never copied into this repository.

In the first notebook cell, replace the repository Dataset slug with the actual slug and set the data root before importing project modules:

```python
import os
import sys
from pathlib import Path

repo_root = Path("/kaggle/input/<uploaded-repository-dataset>")
sys.path.insert(0, str(repo_root))
os.environ["RSNA_KNEE_DATA_ROOT"] = "/kaggle/input/competitions/rsna-knee-abnormality-detection"

from src import config
```

`src.config.DATA_ROOT` auto-detects the two supported Kaggle mount layouts and prefers the `competitions/` form. The environment variable is available for the local subset override described below. After import, modules should read paths and constants from `src.config`; notebook cells should not scatter `/kaggle/input/...` paths throughout the analysis.

## Generating a small local subset

A local subset is for developing and smoke-testing loaders only, not for running the EDA conclusions. To make one without downloading the full competition:

1. Attach the competition to a Kaggle notebook.
2. Select approximately 20–50 exams after the verified exam identifier and directory layout are known.
3. Copy only those exams, including their needed DICOM files and small tabular context, from the read-only competition mount into a directory under `/kaggle/working`.
4. Compress that working directory and download the resulting archive.
5. Extract it locally under the configured `LOCAL_SUBSET_ROOT` directory.

The exact copy command will be added after the Data-tab listing is available; it must use the real identifiers and folders rather than guessed names. Do not run an unrestricted recursive copy or a competition-wide download.

For local loader development, set the override before importing `src.config`:

```python
import os
from pathlib import Path

os.environ["RSNA_KNEE_DATA_ROOT"] = str(Path("local_subset"))
```

## DICOM handling policy

Any future code that touches DICOM pixels must be lazy and per-exam. It must not load the complete dataset, glob the entire DICOM tree, or build an index that requires reading every DICOM file. The synthetic fixture exists only to make pytest runnable in the local authoring environment; it is not an EDA data substitute.

## EDA scope

The EDA notebook will describe exam, series, and slice distributions; MRI sequence and orientation availability; label prevalence and co-occurrence; site-related variation in acquisition and labels; report languages and lengths; representative report examples; and a recommendation for patient/site-aware cross-validation splits.

## Data safety

Do not commit DICOM files, reports, labels, competition downloads, model weights, caches, or credentials.
