# EDA Findings — RSNA Knee Abnormality Detection

Established from direct inspection of the competition data in a Kaggle notebook.
Every number below was measured, not assumed. Inferences are marked as such.

---

## 1. Layout and scale

Data root (note the `competitions/` level):

```
/kaggle/input/competitions/rsna-knee-abnormality-detection/
├── train.csv                 4,407 rows × 14 cols   (5.69 MB)
├── train_series.csv         24,371 rows ×  5 cols   (3.46 MB)
├── test.csv                      3 rows ×  1 col    (placeholder)
├── test_series.csv              15 rows ×  5 cols   (placeholder)
├── sample_submission.csv         3 rows × 13 cols
├── train_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm
└── test_series/   (same layout)
```

- 4,407 exam folders in `train_series/`.
- ~5.5 series per exam (24,371 / 4,407).
- Sampled exam had 5 series of 22, 18, 16, 16, 22 slices. Volumes are small.

`test.csv` has 3 rows and `sample_submission.csv` is 0.5-filled probabilities:
this is a **code competition** with a hidden test set swapped in at scoring time.
The submission notebook must run with internet disabled.

---

## 2. Verified schema

**train.csv**

| column | type | notes |
|---|---|---|
| `StudyInstanceUID` | str | exam key, matches folder name |
| `Report` | str | free text, 9 languages, no language column |
| 12 label columns | float64 | values `{0.0, 1.0, NaN}` |

Label columns, exactly as spelled (note the apostrophe in `Baker's`):

```
ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA,
PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture
```

**train_series.csv / test_series.csv**

| column | type | values |
|---|---|---|
| `StudyInstanceUID` | str | |
| `SeriesInstanceUID` | str | matches folder name |
| `Fluid_Sensitive` | int | 0 / 1 |
| `Fat_Suppression` | int | 0 / 1 |
| `Anatomical_Plane` | str | Sagittal / Axial / Coronal |

No sequence names, no scanner fields, no institution column in the CSVs.

**sample_submission.csv** — `StudyInstanceUID` plus the same 12 columns, probabilities in [0,1].

---

## 3. The defining constraint: 1.3% label coverage

| | exams |
|---|---|
| all 12 labels present | **58** |
| zero labels present | **4,349** |

It is strictly all-or-nothing — no exam is partially labeled.

Positives among the 58:

| label | pos | neg | rate |
|---|---:|---:|---:|
| Effusion | 35 | 23 | 0.603 |
| Synovitis | 27 | 31 | 0.466 |
| Medial Meniscus | 26 | 32 | 0.448 |
| ACL | 24 | 34 | 0.414 |
| Lateral Meniscus | 23 | 35 | 0.397 |
| PF OA | 21 | 37 | 0.362 |
| Contusion | 19 | 39 | 0.328 |
| Fracture | 18 | 40 | 0.310 |
| Medial OA | 15 | 43 | 0.259 |
| Baker's | 12 | 46 | 0.207 |
| Lateral OA | 11 | 47 | 0.190 |
| MCL | 9 | 49 | 0.155 |

**Consequence.** This is not supervised image classification. It is weak
supervision: labels must be derived from the 4,349 reports, then an image model
trained on the derived labels. The 58 are a *validation set for the label
extractor*, not a training set — 9–35 positives per label cannot support
training or reliable cross-validation.

**Reports exist in train.csv and are absent from test.csv.** Report text is a
training-time-only signal. No architecture may consume report text at inference.

---

## 4. Reports

Language detected with `langdetect` on the first 400 characters of each report.

| lang | n | % | median words |
|---|---:|---:|---:|
| en | 1,736 | 39.4 | 182 |
| es | 682 | 15.5 | 104 |
| tr | 546 | 12.4 | 85 |
| hr | 406 | 9.2 | 144 |
| el | 321 | 7.3 | 118 |
| de | 262 | 5.9 | 86 |
| bg | 220 | 5.0 | 146 |
| nl | 153 | 3.5 | 114 |
| fr | 81 | 1.8 | 215 |

Nine languages. Greek and Bulgarian are non-Latin script. No empty reports.

The 58 labeled exams span all nine languages — en 28, es 10, tr 6, hr 4, bg 3,
el 3, nl 2, de 2 — but with only 2–4 exams for de/nl/el/bg, extractor accuracy
can be validated **only in aggregate**, never per language.

### Structure is unreliable

Section-header terms, as a share of all reports:

```
findings 19.5%   conclusion 15.7%   impresión 15.5%   impression 15.3%
technique 13.7%  indication  9.9%   hallazgos   7.9%  resultados  7.6%
técnica   7.6%   klinische   3.5%   bevindingen 3.4%
```

A large fraction of reports carry no section markers at all — one observed
English report is unpunctuated run-on findings. **Do not build section-based
segmentation.** Extraction must handle whole reports.

---

## 5. Text quality

### De-identification placeholders

| token | occurrences |
|---|---:|
| `[DATE]` | 710 (in 594 reports) |
| `[TIME]` | 359 |
| `[REDACTED]` | 110 |
| `[ID]` | 94 |
| `[NAME]` | 38 |
| `[YEAR]` | 14 |
| `[IDENTIFIER]` | 5 |
| `[PROFESSION]` | 2 |
| `[AFFILIATION]` | 1 |

### The `intact` corruption

87 reports, 228 occurrences of `intact` glued directly to a digit. Decoded from
context, the de-ID pipeline substituted the two-character string `0.` with the
word `intact`:

| in the data | actual |
|---|---|
| `A intact3 cm subchondral bone cyst` | `0.3 cm` |
| `1xintact3 cm` | `1x0.3 cm` |
| `intact9xintact4cm` | `0.9x0.4cm` |
| `3.2x1.5xintact9 cm` | `3.2x1.5x0.9 cm` |
| `intact54xintact55 cm` | `0.54x0.55 cm` |

This is dangerous because `intact` is simultaneously the primary English
negation cue ("the ACL is intact"). Normalization must run before any keyword
logic:

```python
text = re.sub(r"intact(?=\d)", "0.", text)
```

The lookahead keeps genuine uses untouched — `intact` never precedes a digit in
real prose. A scan for other word-glued-to-digit tokens found only ordinary
missing-space typos (`grade5` ×5, `CORT1`, `img2`, `angle1`) at 1–5 occurrences.
`intact` is the only systematic corruption.

---

## 6. Site proxy from placeholder signatures

There is no institution column, but de-identification configuration varies by
site and leaves a fingerprint. Which placeholders co-occur in a report,
crossed with language:

| signature | en | nl | de | el | es | bg | hr | tr | fr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `[DATE]+[TIME]` | 247 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `[DATE]+[TIME]+[REDACTED]` | 82 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `[ID]` alone | 43 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `[DATE]` alone | 14 | 151 | 24 | 22 | 13 | 3 | 1 | 0 | 0 |
| none | 1,307 | ~0 | 237 | 298 | 666 | 217 | ~405 | 546 | 81 |

**Reading.** Every `[TIME]`-bearing signature is exclusively English. Dutch is
151/153 `[DATE]`. Turkish, French and Croatian are essentially placeholder-free.

**Why it matters.** English is 39% of the corpus and is certainly several
institutions pooled. Language alone cannot separate them; de-ID signature yields
roughly four distinguishable English groups. This is the best available site
proxy so far — useful as a CV grouping variable, and a warning about what the
model must not learn.

**Limitation.** 1,307 English reports carry no placeholders and remain
unresolved by this method.

---

## 7. Open questions — resolve before writing an extractor

### 7.1 Label definitions are not keyword presence *(highest priority)*

Two of the 58, read against their labels:

**Spanish exam** — positives `[PF OA, Effusion]`
- "Leve derrame articular" (mild joint effusion) → **Effusion = 1**
- "Amputación marginal del cuerpo del menisco lateral" → **Lateral Meniscus = 0**

**English exam** — positives `[Lateral Meniscus, PF OA, Synovitis]`
- "Horizontal tear at anterior horn of the lateral meniscus" → **Lateral Meniscus = 1**
- "Moderate joint effusion, distended suprapatellar bursa" → **Effusion = 0**

The meniscus pair is explicable: the label likely means *tear* specifically, and
marginal amputation is not a tear. The effusion pair is not — mild scores 1
while moderate scores 0.

This is either a strict clinical definition not yet found, or label noise in the
only ground truth available. n=2, so do not over-conclude. Resolve by reading
the competition's published labeling guideline (Data tab / Discussion) and by
auditing all 58 exports in `labeled_58.md`.

### 7.2 DICOM metadata — not yet examined

The remaining EDA branch. On a **sample of a few hundred exams only** (never all
24,371 series), pull `Manufacturer`, `ManufacturerModelName`, `StationName`,
`MagneticFieldStrength`, `InstitutionName` if present, plus pixel spacing and
slice thickness. Goals: complete the site proxy where placeholders fail, and
quantify protocol variation.

### 7.3 Evaluation metric

Probability-valued submission is consistent with an AUC metric, and a separate
efficiency leaderboard is documented externally. Confirm both on the Evaluation
tab.

---

## 8. Design implications

1. **Image-only at inference.** Reports are training-time supervision. Nothing
   in the inference path may touch text.
2. **The pipeline is two-stage.** Report → labels (the hard, high-leverage
   part), then images → labels. Extractor quality caps everything downstream.
3. **NaN is unknown, never zero.** Label parsing must return an observed-mask
   alongside values. Every count reported as positive / negative / missing —
   never a bare mean over a column that is 98.7% NaN.
4. **Negation handling is mandatory** in nine languages. "no evidence of
   meniscal tear" and "meniscal tear" share the keyword.
5. **Group CV** by site proxy (de-ID signature + language + DICOM metadata),
   not randomly.
6. **Lazy, per-exam DICOM access.** No function may glob the whole tree or
   build an index requiring every file to be read.
