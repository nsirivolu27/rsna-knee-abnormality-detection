# Knee MRI Abnormality Detection

An image processing system that reads knee MRI scans and estimates how likely
each of twelve common knee problems is to be present.

Built for the RSNA 2026 challenge on Kaggle, using 4,407 real knee examinations
collected from around nineteen hospitals worldwide.

This is a research scaffold, not a clinical tool. It does not establish
diagnostic accuracy or suitability for patient care.

---

## What it looks for

Twelve findings, scored independently for every scan:

| | |
|---|---|
| **Ligaments** | ACL tear, MCL tear |
| **Cartilage discs** | Medial meniscus tear, lateral meniscus tear |
| **Joint wear** | Medial, lateral, and kneecap (patellofemoral) osteoarthritis |
| **Inflammation** | Joint effusion, synovitis, Baker's cyst |
| **Bone injury** | Bone bruise (contusion), fracture |

The output is a probability between 0 and 1 for each one, not a yes or no
answer. A radiologist reading the same scan would talk in terms of confidence
too.

## How the system works

An MRI examination is not a single picture. Each exam in this dataset contains
about five and a half separate scan series, and each series is a stack of
sixteen to twenty two cross sectional slices through the knee. Different series
are taken in different orientations and with different settings, and each one
makes certain tissues stand out.

The pipeline moves a scan through six stages. Five are built.

```
  ┌──────────┐   ┌───────┐   ┌───────────┐   ┌───────┐   ┌─────────┐   ┌────────────┐
  │  1 READ  │──▶│ 2 ROUTE│──▶│ 3 NORMALIZE│──▶│ 4 CACHE│──▶│ 5 PREDICT│──▶│ 6 SUBMIT   │
  │  DICOM   │   │ 4 slots│   │ fixed shape│   │  .npz  │   │  model   │   │submission  │
  │ → volume │   │ per exam│  │ + rescale  │   │        │   │          │   │   .csv     │
  └──────────┘   └───────┘   └───────────┘   └───────┘   └─────────┘   └────────────┘
   dicom_io      series_       preprocessing    cache       inference      submission
                 selection                                     ▲
                                                               │
                                              ┌────────────────┴────────────────┐
                                              │  TRAINING LABELS (not yet run)  │
                                              │  reports → labeling.py →        │
                                              │  soft_labels → agreement        │
                                              └─────────────────────────────────┘
```

**1. Read.** Medical scanners save images in a format called DICOM, one file per
slice, along with technical details about how the scan was taken. The system
loads a stack of these into a single 3D volume and records the settings that
came with it. All 4,407 exams read in about seventy six seconds with no
failures.

**2. Route.** Not every series is useful for every finding. A fluid sensitive
sagittal scan makes joint fluid glow bright and shows the meniscus clearly. A
coronal scan is the one that shows the inner and outer compartments side by
side. The system picks four fixed viewpoints per exam so the model always sees
the same kinds of images.

| Viewpoint | Available in |
|---|---:|
| Axial, fluid sensitive | 100.0% of exams |
| Sagittal T1 | 96.8% |
| Coronal, fluid sensitive | 96.4% |
| Sagittal, fluid sensitive | 94.2% |

When a viewpoint is missing it is marked as missing rather than filled with
blank images, so the model can tell the difference between "nothing there" and
"we did not look".

**3. Normalize.** Real hospital scans vary enormously. Different machines,
magnet strengths, slice thicknesses, and brightness scales. MRI has no absolute
brightness units at all, so the same tissue can appear at completely different
values on two machines. This stage resizes every volume to a common size, pads
or trims to a fixed number of slices, and rescales brightness so scans from
different hospitals become comparable.

**4. Cache.** Preprocessing is the expensive part, so the result is written to
disk keyed by the exam, the series chosen, and the exact settings used. Change
any setting and the cache key changes, so stale volumes can never be silently
reused.

**5. Predict.** A predictor is handed one prepared exam and returns twelve
probabilities. The predictor is passed in rather than hard wired, which keeps
the model choice out of the pipeline and lets the whole path be tested without
one. If a single exam cannot be read, or the model raises on it, that exam falls
back to a default and the run continues. A crash partway through costs the whole
submission; one fallback row costs one exam.

**6. Submit.** The answer file has an unforgiving format: exact column names in
exact order, one row per test exam, every value a finite probability. That
contract lives in one module and is checked before anything is written.

There is no trained model yet, so the current predictor returns each label's
base rate. It cannot rank exams and therefore scores at chance, which is the
honest result for a pipeline check.

## The interesting problem

Only 58 of the 4,407 exams come with expert labels. The other 4,349 have no
labels at all.

What they do have is the original radiology report, written by the doctor who
read the scan at the time, in one of nine languages. So the plan is to read
training labels out of those reports and use them to teach the model.

That turns out to be harder than it sounds. The expert labels were assigned by a
panel looking at the images, not by reading the reports, and the two sometimes
disagree outright. One exam is labeled as having a Baker's cyst in a report that
says there is no cyst. Another says "mild joint effusion" and is labeled
positive, while a different exam says "moderate joint effusion" and is labeled
negative.

So a large part of this project is measuring how much of the truth actually
survives in the written reports, finding by finding, and being honest about
which of the twelve can be learned this way and which cannot.

Because a rule based reader cannot solve this, the repository contains no
keyword, section, or negation extractor by design. Labels are produced
externally as probabilities and then scored against the 58 expert exams.

## Keeping the model honest

Scans from the same hospital look alike. Same scanner, same settings, same
habits. A model can learn to recognize the hospital instead of the injury, and
then score well in testing while being useless on a new hospital's scans.

Two safeguards are built in:

- Scanner manufacturer, model, magnet strength and scan descriptions are used
  only to decide which images to feed the model. They are never given to the
  model itself.
- Validation splits are grouped so that every scan from a given hospital and
  language lands in the same split. The five folds hold 869, 865, 870, 880 and
  865 exams, and no group crosses a fold. The 58 expert exams sit outside the
  training folds entirely.

This is conservative internal validation. It does not establish that the system
would work at a hospital outside this dataset.

## Where the project stands

| Stage | Status |
|---|---|
| 1. Reading DICOM and scan settings | Working. All 4,407 exams, 76 seconds, no failures |
| 2. Choosing which series to use | Working, coverage measured |
| 3. Normalizing volumes | Working |
| 4. Caching | Working |
| 5. Prediction loop | Working, with a placeholder predictor |
| 6. Submission file | Working, validated |
| Grouped validation splits | Working, zero leakage |
| Deriving labels from reports | Runner built, not yet run at scale |
| Trained model | Not started |

## Running it

The full dataset is 247 GB and is never downloaded. Everything runs against
Kaggle's read only copy, which is mounted at either of two paths depending on
how the data was attached. The code handles both.

```python
# In a Kaggle notebook with the competition data attached, Internet ON
!git clone -q https://github.com/nsirivolu27/rsna-knee-abnormality-detection.git /kaggle/working/repo
import sys; sys.path.insert(0, "/kaggle/working/repo")

import pandas as pd
from src import config, dicom_io, site_proxy, splits

scans = dicom_io.exam_metadata(cache_path="/kaggle/working/exam_metadata.parquet")
groups = site_proxy.build_site_proxy(series_metadata=scans,
                                     reports=pd.read_csv(config.TRAIN_CSV))
folds = splits.build_grouped_folds(groups, n_splits=5, seed=42)
```

Tests run anywhere, with no scan data present:

```bash
pytest
```

## Producing a submission

From a Kaggle notebook with the competition data attached and **internet
disabled**:

```python
import sys; sys.path.insert(0, "/kaggle/input/<your-repo-dataset>")
from scripts.run_submission import main

submission, diagnostics = main()
```

It reads the test exam list, runs every exam through stages 1 to 5, checks the
result against the submission contract, and writes `submission.csv`. The
diagnostics frame reports, per exam, whether images were read and how many of
the four viewpoints were found.

Because the submission notebook cannot reach the internet, the repository has to
be attached as a Kaggle Dataset rather than cloned.

## Repository map

```
src/
  dicom_io.py          reads scan files and their settings
  series_selection.py  picks which scans to use
  preprocessing.py     resizing, slice handling, brightness normalization
  cache.py             stores processed volumes
  dataset.py           feeds volumes to the model
  inference.py         runs the prediction loop over exams
  submission.py        builds and checks the answer file
  reports.py           cleans up radiology report text
  labels.py            loads the 58 expert labeled exams
  labeling.py          runs an external model over reports, resumably
  soft_labels.py       loads and validates labels derived from reports
  agreement.py         measures derived labels against the expert ones
  site_proxy.py        works out which hospital a scan probably came from
  splits.py            builds validation splits that respect hospitals
  data.py              exam index joining the tables and folders
  config.py            paths, label names, seed
scripts/run_labeling.py    entry point for the labeling run
scripts/run_submission.py  entry point for producing a submission
prompts/                 versioned label extraction templates
docs/                    research brief and measured findings
tests/                   run without any scan data
```

## Documentation

- [`docs/RESEARCH_BRIEF.md`](docs/RESEARCH_BRIEF.md) the methods first framing
- [`docs/EDA_FINDINGS.md`](docs/EDA_FINDINGS.md) everything measured about the
  dataset, and the source of every number quoted above

## Scope

This is a research competition entry, not a medical device. Nothing here is
validated for patient care and no output should be read as a diagnosis. The
scans are de identified and stay on Kaggle's platform under the competition
terms. Report text is never copied into this repository.
