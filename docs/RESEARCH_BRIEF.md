# RSNA Knee Abnormality Detection — Research Brief

> **Problem framing:** How can we build a reliable image-only system for twelve knee abnormalities when almost all exams lack expert labels, clinical reports disagree with image-derived ground truth, and the MRI corpus is too large to move locally?

This brief turns the verified EDA into a research problem, an evidence base, and an implementation boundary. It is intentionally not a model proposal or a claim of leaderboard performance.

## 1. The problem we are actually solving

The apparent task is multi-label MRI classification. The measured task is different: construct a defensible weak-supervision pipeline for 4,349 exams whose twelve labels are unknown, while preserving a small expert-labeled partition for validation and preventing acquisition-site leakage.

The system must answer three linked questions:

1. **What can supervise training?** Reports exist for training exams, but labels were assigned from images and report text contains contradictions.
2. **Which image evidence should enter the model?** The series table supplies routing flags, but each exam has multiple sequences and nonuniform slot availability.
3. **How do we measure generalization honestly?** Scanner and language proxies cluster exams, so random folds risk learning acquisition identity instead of pathology.

## 2. Verified evidence

| Finding | Measurement | Design consequence |
|---|---:|---|
| Training exams | 4,407 | Keep the full corpus indexed without copying DICOM locally |
| Fully labeled exams | 58 | Reserve as expert validation; do not train on them |
| Training-eligible exams | 4,349 | Use reports only for external weak-label generation |
| Report languages | 9 | Do not assume an English-only extractor |
| Site-proxy groups | 24 | Group folds by measured proxy, not random exam ID |
| Forced residual merges | 9 | Preserve merge audit columns and inspect group behavior |
| DICOM metadata reads | 4,407 exams, 0 unreadable | Lazy per-exam access is viable |

### Canonical image-slot coverage

| Slot | Selected | Missing | Coverage |
|---|---:|---:|---:|
| Sagittal fluid-sensitive | 4,150 | 257 | 94.1684% |
| Coronal fluid-sensitive | 4,248 | 159 | 96.3921% |
| Axial fluid-sensitive | 4,407 | 0 | 100.0000% |
| Sagittal T1 | 4,266 | 141 | 96.8005% |

The missingness is part of the data contract. A missing sequence is not a negative finding, so every slot is paired with a presence mask.

## 3. Research hypotheses

- **H1 — Soft labels can unlock image training:** an external LLM can turn reports into probability-valued supervision, but agreement must be measured against the 58 expert exams before those labels are trusted.
- **H2 — Image routing should be metadata-light:** the provided plane/fluid/fat-suppression flags can select canonical inputs without exposing scanner or institution identity to the model.
- **H3 — Site-aware validation is necessary:** measured scanner/language proxies are strong enough that random folds would overstate generalization.
- **H4 — Explicit missingness is safer than forced completion:** slot masks allow downstream work to distinguish an absent sequence from a low-signal image or a negative label.

## 4. Proposed evidence-preserving pipeline

```text
train.csv reports ──external LLM──> soft labels ──agreement check──┐
                                                                  │
train_series.csv ──metadata routing──> four canonical slots ──────┼──> image-only training
DICOM series ──lazy per-exam load──> normalize/sample/resize ─────┘

site proxies ──grouped folds──> leakage-aware evaluation
presence masks ───────────────> explicit sequence missingness
```

The repository currently implements the evidence-preserving front half: verified loading, lazy DICOM access, deterministic routing, fixed-shape preprocessing, caching, label masks, external soft-label validation, and grouped splits. Architectures, training loops, benchmarks, inference, and submissions remain deliberately out of scope.

## 5. What the research rules out

- No deterministic report keyword or negation extractor: labels are image-derived and the reports contain direct reversals.
- No report text at inference: `test.csv` has no reports, and the intended downstream system is image-only.
- No random cross-validation: acquisition proxies can create deceptively easy folds.
- No scanner metadata as model features: manufacturer, model, field strength, and descriptions are routing/leakage signals, not pathology evidence.
- No conversion of `NaN` to zero: unknown labels remain unknown and carry an observed mask.
- No full-tree DICOM indexing: access is bounded to the requested exam and selected series.

## 6. Next research gates

1. Validate the external soft-label generator against all 58 expert exams with per-label AUC, uncertainty intervals, bias, and macro-AUC.
2. Run the canonical loader on the 20–50 exam local subset, including incomplete-slot and malformed-DICOM cases.
3. Inspect cache reproducibility and memory behavior before introducing any model framework.
4. Only then choose a training design that consumes image tensors, slot presence masks, and soft-label observation masks.

## 7. Source of truth

- [Verified EDA findings](EDA_FINDINGS.md)
- [Canonical series selector](../src/series_selection.py)
- [Preprocessing contract](../src/preprocessing.py)
- [Lazy exam dataset](../src/dataset.py)
