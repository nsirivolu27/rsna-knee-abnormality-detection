# RSNA Knee Abnormality Detection

Scaffold and exploratory data analysis for the Kaggle RSNA Knee Abnormality Detection competition. This repository is intentionally limited to data inspection, DICOM/report loading utilities, label parsing, plotting helpers, and reproducible EDA. It does not contain model architectures, training loops, benchmarks, or submission-writing code.

## Current status

The repository is in the scaffold phase. The competition schema and data layout are **unverified** until the competition owner provides the Data-tab file/folder listing, each CSV header plus three sample rows, and the exact twelve label names. No parser should guess those values.

## Planned layout

`src/config.py` will be the single source of truth for paths, constants, and the verified schema. The remaining files will be added in later scoped chunks.

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

## Local setup

The dependency list uses minimum-version constraints rather than exact pins because Kaggle notebooks run a fixed image and hard pins can create conflicts. The standard Kaggle image is expected to provide NumPy, pandas, Matplotlib, seaborn, and pydicom. pytest and nbstripout are local/Replit authoring tools used for tests and notebook hygiene.

For local smoke tests after the test files are added:

```bash
pip install -r requirements.txt
pytest
```

## Notebook hygiene

The committed EDA notebook must have all outputs cleared. Before committing notebook changes, run:

```bash
nbstripout notebooks/01_eda.ipynb
```

The written EDA takeaways will also be maintained in `notebooks/eda_findings.md` so conclusions are reviewable in a normal text diff.

## Using the package on Kaggle

The cleanest workflow is to publish this repository as a Kaggle Dataset containing the source tree, then attach that dataset to the competition notebook. The competition data remains a separate attached dataset and is never copied into this repository.

In the first notebook cell, replace both placeholder dataset slugs with the actual Kaggle Dataset slugs:

```python
import os
import sys
from pathlib import Path

repo_root = Path("/kaggle/input/<uploaded-repository-dataset>")
sys.path.insert(0, str(repo_root))
os.environ["RSNA_KNEE_DATA_ROOT"] = "/kaggle/input/<competition-data-dataset>"

from src import config
```

After that, modules should import paths and constants from `src.config`; notebook cells should not scatter `/kaggle/input/...` paths throughout the analysis. The exact data subdirectories and CSV filenames will be added to the config only after the verified Data-tab listing is available.

## EDA scope

The EDA notebook will describe exam, series, and slice distributions; MRI sequence and orientation availability; label prevalence and co-occurrence; site-related variation in acquisition and labels; report languages and lengths; representative report examples; and a recommendation for patient/site-aware cross-validation splits.

## Data safety

Do not commit DICOM files, reports, labels, competition downloads, model weights, caches, or credentials.
