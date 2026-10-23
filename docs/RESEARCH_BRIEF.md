# RSNA Knee Abnormality Detection — Research Protocol Brief

## Study question

**Among knee MRI examinations, can an image-only multi-label learning system identify twelve image-derived abnormalities when direct expert labels are available for only a small validation partition and the remaining exams require report-derived weak supervision?**

This is a methodological study of weak supervision, sequence selection, and leakage-aware evaluation. It is **not** a clinical diagnostic validation study, a prospective study, or evidence of clinical utility.

## Clinical and methodological rationale

The competition is superficially framed as multi-label classification. The verified data structure creates a more specific research problem:

- The corpus contains **4,407 examinations**, but only **58 examinations have all twelve expert labels**.
- The remaining **4,349 examinations are training-eligible but have unknown labels**.
- The twelve labels were assigned from images. Reports are available during training but absent from the test input and contain documented contradictions with the image-derived labels.
- Each examination contains multiple MRI series, with measurable variation in the availability of canonical sequences.
- Scanner and language-related variables form acquisition proxies that can cluster examinations and create optimistic random-split estimates.

The central question is therefore not simply whether a model can fit images. It is whether an evidence-preserving pipeline can create and evaluate image supervision without confusing report language, acquisition site, or sequence availability with pathology.

## Aims

### Primary methodological aim

To establish a reproducible pipeline for image-only modeling that preserves uncertainty in weak labels, represents sequence missingness explicitly, and prevents measured acquisition-proxy groups from crossing validation folds.

### Secondary aims

1. Quantify agreement between externally generated report-derived soft labels and the 58-examination expert partition.
2. Measure the availability of four canonical MRI input slots using only the verified series-routing fields.
3. Define a lazy, fixed-shape DICOM loading path that does not require copying or indexing the approximately 247 GB corpus locally.
4. Identify limitations that must be resolved before any model performance is interpreted as clinically meaningful.

## Data provenance and unit of analysis

The unit of analysis is the **examination**, identified by `StudyInstanceUID`. The verified competition files contain:

- `train.csv`: 4,407 examination rows and twelve label columns.
- `train_series.csv`: 24,371 series rows with `StudyInstanceUID`, `SeriesInstanceUID`, `Fluid_Sensitive`, `Fat_Suppression`, and `Anatomical_Plane`.
- DICOM files organized by examination, series, and instance identifiers.
- Reports in `train.csv`; reports are absent from `test.csv`.

The full corpus remains mounted in Kaggle. It is not copied into this repository or into a local computer. The repository contains utilities and synthetic fixtures, not competition DICOM or report data.

## Label provenance

The twelve labels are:

`ACL`, `MCL`, `Medial Meniscus`, `Lateral Meniscus`, `Medial OA`, `Lateral OA`, `PF OA`, `Effusion`, `Synovitis`, `Baker's`, `Contusion`, and `Fracture`.

The 58 complete rows reproduce all twelve known positive counts exactly. They are treated as an **expert validation partition**, not as a training sample. The other 4,349 rows are not treated as confirmed negatives; their label state is unknown.

This distinction is essential. Reports are external weak-supervision material, not a replacement reference standard. The planned supervision sequence is:

1. An external LLM produces probability-valued labels from reports.
2. Those soft labels are evaluated against the 58 expert examinations.
3. An image model may use the accepted soft labels during training.
4. Inference uses images only; reports are not an input.

No deterministic keyword, section, or negation extractor is part of the design. Direct report-versus-label reversals show that a text rule cannot be assumed to recover the image-derived reference labels.

## Image input definition

Four canonical slots are routed from `train_series.csv`:

| Slot | Selection rule | Full-corpus coverage |
|---|---|---:|
| Sagittal fluid-sensitive | Sagittal + fluid-sensitive; prefer fat suppression | 4,150 / 4,407 (94.1684%) |
| Coronal fluid-sensitive | Coronal + fluid-sensitive; prefer fat suppression | 4,248 / 4,407 (96.3921%) |
| Axial fluid-sensitive | Axial + fluid-sensitive; prefer fat suppression | 4,407 / 4,407 (100.0000%) |
| Sagittal T1 | Sagittal + non-fluid-sensitive | 4,266 / 4,407 (96.8005%) |

A missing slot is not interpreted as a negative finding. The dataset contract therefore returns a per-slot presence mask. The selector does not expose manufacturer, scanner model, field strength, station, institution, or `SeriesDescription` to the image-facing interface.

`train_series.csv` has no slice-count field. The selector accepts an optional caller-derived `slice_count` only when available; otherwise all candidates tie at that criterion and lexicographic `SeriesInstanceUID` ordering is used. Coverage measurement does not perform winner selection or tie-breaking.

Preprocessing is deliberately model-agnostic: per-volume robust intensity scaling, deterministic depth sampling, and bilinear in-plane resizing produce fixed-shape `float32` arrays. Missing slots use explicit zero placeholders together with the presence mask; they are not treated as negative evidence.

## Leakage-aware validation plan

A site proxy is not a confirmed institution identifier. It is a measured grouping construct derived from available DICOM metadata, normalized manufacturer/model information, rounded field strength, report-language information, and report-placeholder signatures. The full-corpus audit produced **24 proxy groups**, with **9 deterministic residual merges** recorded in audit columns.

The planned five-fold assignment has fold sizes of **869, 865, 870, 880, and 865** among training-eligible examinations. The 58 expert examinations remain auditable but have no training fold. The grouping check passed: no proxy group crosses folds.

This design estimates performance under the observed acquisition grouping more conservatively than random examination-level splitting. It does not establish generalization to an external hospital, scanner vendor, population, or clinical workflow.

## Prespecified weak-label evaluation

Agreement between report-derived probabilities and expert labels will be summarized by:

- Per-label ROC-AUC with a seeded 1,000-resample 95% bootstrap interval.
- Macro-AUC across labels with available evaluation support.
- Positive-rate bias relative to the expert partition.
- Threshold metrics explicitly identified as same-sample, optimistic upper bounds rather than independent clinical performance estimates.
- Separate accounting for parse failures; failures remain `NaN` and are never replaced with 0.5.

The small expert partition contains only 9–35 positives per label. Consequently, close AUC values should not be interpreted as reliably different without considering interval width and label prevalence.

## What can and cannot be concluded

### Supported by the current work

- The verified schema and label counts are internally consistent.
- Lazy, per-examination DICOM access is feasible; the full metadata pass read 4,407 exams with zero unreadable exams.
- Canonical sequence availability is measurable and nonuniform.
- Acquisition-proxy grouping changes the validation design and has been audited.
- Unknown labels and missing sequences can be represented explicitly rather than coerced into negatives.

### Not established by the current work

- Diagnostic accuracy, sensitivity, specificity, calibration, clinical utility, or reader agreement.
- External validity beyond this competition cohort.
- Whether report-derived soft labels improve image-model performance.
- Whether the site proxy adequately captures all acquisition dependence.
- Whether any model would be safe or appropriate for clinical use.

## Limitations and threats to validity

1. **Reference-label scarcity:** only 58 examinations have complete expert labels, limiting precision for per-label agreement estimates.
2. **Weak-label uncertainty:** reports are heterogeneous, multilingual, and sometimes contradict the image-derived labels.
3. **Proxy uncertainty:** the grouping key approximates acquisition site; it is not a verified institution identifier.
4. **Spectrum and sampling:** the competition cohort may not represent the prevalence, protocols, or patient mix of a clinical service.
5. **No external validation:** all current findings are internal to the competition data.
6. **No clinical endpoint:** the labels are examination-level abnormalities, not patient outcomes or treatment decisions.
7. **Unresolved guideline question:** the competition labeling guideline should be reviewed before making clinical interpretations of individual labels.

## Research gates before modeling claims

1. Complete soft-label agreement analysis on all 58 expert examinations.
2. Test the loader on a 20–50 examination local subset, including missing slots and malformed-DICOM cases.
3. Verify preprocessing and cache reproducibility, memory usage, and fold isolation.
4. Define an analysis plan for missingness, label support, and uncertainty before selecting a model framework.
5. Report any downstream model only with the expert/weak-label distinction, grouped validation scheme, and limitations above.

## Implemented repository boundary

The repository currently provides verified data loading, lazy DICOM utilities, deterministic canonical routing, model-agnostic preprocessing, versioned caching, explicit image and label masks, external soft-label validation, and grouped folds. Model architectures, training loops, benchmark claims, inference, and submission generation remain outside this research brief.

See [EDA findings](EDA_FINDINGS.md) for the measured source record and the implementation references in [the README](../README.md).
