# RSNA Knee Abnormality Detection

## Research protocol and reproducible data scaffold

This repository develops the data and evaluation foundation for studying **image-only multi-label recognition of twelve knee MRI abnormalities under weak supervision**.

The central methodological question is:

> **Can an image-only learning system identify twelve image-derived abnormalities when complete expert labels are available for only a small validation partition, reports provide imperfect weak supervision, and acquisition structure can create validation leakage?**

This repository is a research scaffold, not a clinical diagnostic system. It does not establish diagnostic accuracy, clinical utility, calibration, external validity, or suitability for patient care.

## Research framing

Read the full methods-first framing in the [Research Protocol Brief](docs/RESEARCH_BRIEF.md), or view the [standalone HTML presentation](docs/RESEARCH_BRIEF.html).

The authoritative measured findings are maintained in [docs/EDA_FINDINGS.md](docs/EDA_FINDINGS.md). Every reported dataset number in that document was measured in Kaggle against the attached competition data; it is not inferred from the repository fixtures.

## Verified study setting

| Quantity | Verified finding | Methodological implication |
|---|---:|---|
| Training examinations | 4,407 | The unit of analysis is `StudyInstanceUID` |
| Fully expert-labeled examinations | 58 | Reserved for validation of report-derived supervision |
| Training-eligible examinations | 4,349 | Unknown labels, not confirmed negatives |
| Series rows | 24,371 | Multiple MRI series require deterministic routing |
| Report languages | 9 | English-only text assumptions are not defensible |
| Site-proxy groups | 24 | Validation must respect measured acquisition structure |
| Forced residual group merges | 9 | Coarsening decisions are retained in audit columns |
| DICOM metadata read | 4,407 exams; 0 unreadable | Lazy, per-examination access is feasible |

The competition corpus is approximately 247 GB decompressed. It remains mounted in Kaggle and is not copied into this repository or onto a local computer.

## Label provenance and weak supervision

The twelve labels are:

```text
ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA,
PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture
```

The 58 complete label rows reproduce all twelve known positive counts. They are treated as an **expert validation partition**, not as a training set. The other 4,349 examinations have unknown labels; `NaN` is preserved as unknown and never converted to zero.

Reports are available during training but absent from the test input. They are treated as external weak-supervision material only:

1. An external LLM produces probability-valued labels from reports.
2. Those soft labels are evaluated against the 58 expert examinations.
3. An image model may use accepted soft labels during training.
4. Inference uses images only.

The repository intentionally contains no deterministic report keyword, section, negation, or rule-based extractor. The verified EDA documents direct report-versus-label reversals, so report text cannot be assumed to reproduce the image-derived labels.

## Image input contract

The series table provides routing fields but no sequence names or scanner fields. The scaffold selects at most one series for each canonical slot:

| Canonical slot | Routing rule | Full-corpus coverage |
|---|---|---:|
| Sagittal fluid-sensitive | Sagittal + fluid-sensitive; prefer fat suppression | 4,150 / 4,407 (94.1684%) |
| Coronal fluid-sensitive | Coronal + fluid-sensitive; prefer fat suppression | 4,248 / 4,407 (96.3921%) |
| Axial fluid-sensitive | Axial + fluid-sensitive; prefer fat suppression | 4,407 / 4,407 (100.0000%) |
| Sagittal T1 | Sagittal + non-fluid-sensitive | 4,266 / 4,407 (96.8005%) |

Missing slots are represented by an explicit `presence_mask`. Missingness is not negative evidence.

`train_series.csv` has five columns and no slice-count field. The selector accepts an optional caller-derived `slice_count`; when it is absent, the slice-count criterion is unavailable and lexicographic `SeriesInstanceUID` ordering resolves ties. Coverage measurement does not perform tie-breaking.

The image path does not expose manufacturer, scanner model, field strength, station, institution, or `SeriesDescription` as model inputs. These variables may be useful for routing or leakage analysis, but they are not pathology features in this scaffold.

## Leakage-aware validation

A site proxy is a measured grouping construct, not a confirmed institution identifier. It combines available DICOM and report-derived signals, including normalized manufacturer/model information, rounded field strength, report language, and placeholder signatures.

The verified five-fold assignment contains **869, 865, 870, 880, and 865 training-eligible examinations**. The 58 expert examinations remain auditable but have no training fold. No site-proxy group crosses folds.

This is a conservative internal validation design. It does not establish generalization to another hospital, scanner, patient population, or clinical workflow.

## Repository scope

Implemented:

- Verified schema and path handling for both supported Kaggle mount layouts
- Lazy, per-examination DICOM loading with structured warnings
- Deterministic canonical series selection
- Fixed-shape, model-agnostic preprocessing
- Versioned atomic caching of preprocessed exam tensors
- Framework-neutral lazy exam dataset returning image and missingness masks
- Label loading that preserves `NaN` and exposes observed masks
- External soft-label parsing, checkpointing, provenance, and agreement evaluation
- Acquisition-proxy construction and grouped folds
- Synthetic DICOM fixtures and pytest coverage

Deliberately out of scope:

- Model architectures
- Training loops
- Benchmark or leaderboard claims
- Inference and submission generation
- Clinical deployment or clinical decision support
- Rule-based report extraction

## Reproducibility workflow

### Kaggle data access

The competition data is expected to be attached in Kaggle. The code supports both observed mount layouts:

```text
/kaggle/input/competitions/rsna-knee-abnormality-detection
/kaggle/input/rsna-knee-abnormality-detection
```

Set the root before importing project modules when using an uploaded repository Dataset:

```python
import os
import sys
from pathlib import Path

repo_root = Path("/kaggle/input/<uploaded-repository-dataset>")
sys.path.insert(0, str(repo_root))
os.environ["RSNA_KNEE_DATA_ROOT"] = "/kaggle/input/competitions/rsna-knee-abnormality-detection"

from src import config
```

`src.config` auto-detects the supported roots and centralizes all paths and schema constants.

### Local tests

The standard authoring workflow is:

```bash
pip install -r requirements.txt
pytest
```

The synthetic fixture is for loader development only. It is not a substitute for the Kaggle EDA data.

### Local subset

For loader development, export only approximately 20–50 selected examinations from Kaggle, including their needed DICOM files and small tabular context. Extract the subset under `local_subset/` and set:

```python
import os
from pathlib import Path

os.environ["RSNA_KNEE_DATA_ROOT"] = str(Path("local_subset"))
```

Never perform an unrestricted recursive copy of the competition tree.

### Notebook hygiene

The committed EDA notebook must be output-free:

```bash
nbstripout notebooks/01_eda.ipynb
```

Written findings belong in `notebooks/eda_findings.md` and the authoritative `docs/EDA_FINDINGS.md` so conclusions remain reviewable in ordinary text diffs.

## Data governance

Do not commit competition DICOM files, report text, generated labels, model weights, caches, credentials, or downloaded competition archives. The repository is intended to contain methods, utilities, tests, and measured findings—not redistributed competition data.
