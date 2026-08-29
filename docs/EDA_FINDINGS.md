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

## 6. Site identity — no direct field, four proxies

`StationName` and `InstitutionName` are **fully stripped**: `None` in all 1,088
sampled series. There is no institution column anywhere in this dataset. Site
must be approximated.

Measured on a 200-exam / 1,088-series random sample.

### 6.1 Scanner model — strongest signal

For every non-English language, model is close to a site identifier:

| model | language concentration |
|---|---|
| MAGNETOM Avanto fit | 14/14 German |
| MAGNETOM Prisma | 6/6 Greek |
| SonataVision | 8/8 Greek (a second Greek site) |
| Skyra | 9/9 Spanish |
| MAGNETOM Aera | 5/5 French |
| SIGNA Architect | 5/5 Turkish |
| Ingenia | shared — bg 11, en 13, es 7, hr 9 |

Ingenia is the exception; it is a common scanner and spans four languages.

### 6.2 Manufacturer string spelling

Vendor strings are **not normalized in the source data**, and the specific
spelling is itself informative — it is emitted by the scanner's software
version, so it separates sites and equipment generations:

```
Siemens Healthineers 278 | SIEMENS 233 | Siemens 17
Philips Medical Systems 179 | Philips 138 | Philips Healthcare 15
GE MEDICAL SYSTEMS 189 | GEHC 5
TOSHIBA 27 | CANON_MEC 7      (same vendor lineage; Canon acquired Toshiba Medical)
```

Keep the raw string **and** a normalized vendor column. Discarding the raw
spelling throws away site signal.

### 6.3 SeriesDescription naming convention

House style is site-specific:

| example | convention |
|---|---|
| `pd_tse_fs_sag_320`, `t1_tse_cor_320` | Siemens lowercase-underscore |
| `PDW_TSE_SPAIR_Sag`, `T1W_TSE_Cor` | Philips |
| `COR T1` | terse manual naming |

These encode sequence type, plane (`tra` = transverse = axial), fat suppression
(`fs`, `SPAIR`, `we`) and acquisition matrix (320 / 384 / 448).

`DummySeriesDesc!` appears in **163 of 1,088 series (15%)** — a de-identification
placeholder applied by some contributing sites and not others, and therefore a
fingerprint in its own right. A further 28 are `None`.

### 6.4 Report de-ID placeholder signature

Which placeholders co-occur in a report, crossed with language:

| signature | en | nl | de | el | es | bg | hr | tr | fr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `[DATE]+[TIME]` | 247 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `[DATE]+[TIME]+[REDACTED]` | 82 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `[ID]` alone | 43 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `[DATE]` alone | 14 | 151 | 24 | 22 | 13 | 3 | 1 | 0 | 0 |
| none | 1,307 | ~0 | 237 | 298 | 666 | 217 | ~405 | 546 | 81 |

Every `[TIME]`-bearing signature is exclusively English. Dutch is 151/153
`[DATE]`. Turkish, French and Croatian are effectively placeholder-free. This is
the only proxy that separates English sub-sites.

### 6.5 Recommended grouping key

**38 distinct (manufacturer, model, field strength) combinations** in the
200-exam sample. External sources describe 16–19 contributing sites, so several
sites operate more than one scanner. Grouping by scanner is *finer* than
grouping by site — the conservative direction for cross-validation.

Proposed CV group key:

```
(normalized_manufacturer, model, MagneticFieldStrength,
 detected_language, placeholder_signature)
```

At ~4,400 exams over ~40 groups that averages ~110 exams per group, which
supports 5-fold grouped splits.

**Known weakness.** English exams spread thinly across 19 models with mostly
empty placeholder signatures. English site attribution remains partial and
should be stated as a limitation in any CV write-up.

---

## 7. Acquisition protocol

| field strength | series |
|---|---:|
| 1.5 T | 632 |
| 3.0 T | 416 |
| missing | 40 |

Slice thickness is concentrated at 3.0 mm (677), 4.0 mm (174), 3.5 mm (89) and
2.5 mm (54), with a thin tail down to 0.6 mm. Note that 0.6 appears as two
separate value-counts entries — float representation differs between files, so
any dedupe or grouping on thickness must round first.

Zero read warnings across 1,088 sampled series: the DICOM files are well formed.

---

## 8. Open questions

### 8.1 Label definitions are not keyword presence *(the blocker)*

Two of the 58, read against their labels:

**Spanish exam** — positives `[PF OA, Effusion]`
- "Leve derrame articular" (mild joint effusion) → **Effusion = 1**
- "Amputación marginal del cuerpo del menisco lateral" → **Lateral Meniscus = 0**

**English exam** — positives `[Lateral Meniscus, PF OA, Synovitis]`
- "Horizontal tear at anterior horn of the lateral meniscus" → **Lateral Meniscus = 1**
- "Moderate joint effusion, distended suprapatellar bursa" → **Effusion = 0**

The meniscus pair is explicable — the label likely means *tear* specifically,
and marginal amputation is not a tear. The effusion pair is not: mild scores 1
while moderate scores 0.

Either a strict clinical definition not yet found, or label noise in the only
ground truth available. n=2, so do not over-conclude. Resolve by reading the
competition's published labeling guideline (Data tab / Discussion) and auditing
all 58 in `labeled_58.md`.

**Nothing downstream of the extractor is safe to build until this is settled.**

### 8.2 Evaluation metric

Probability-valued submission is consistent with an AUC metric, and a separate
efficiency leaderboard is documented externally. Confirm both on the Evaluation
tab.

---

## 9. Design implications

1. **Image-only at inference.** Reports are training-time supervision. Nothing
   in the inference path may touch text.
2. **The pipeline is two-stage.** Report → labels (the hard, high-leverage
   part), then images → labels. Extractor quality caps everything downstream.
3. **NaN is unknown, never zero.** Label parsing returns an observed-mask
   alongside values. Every count reported as positive / negative / missing —
   never a bare mean over a column that is 98.7% NaN.
4. **Negation handling is mandatory** in nine languages. "no evidence of
   meniscal tear" and "meniscal tear" share the keyword. English alone contains
   2,975 standalone uses of "intact".
5. **Reports contain ordinary dictation typos** ("less thab 50%") in all nine
   languages. Exact-match extraction will be brittle.
6. **Group CV** by the section 6.5 key, never randomly.
7. **`SeriesDescription` is for routing, not features.** It reliably encodes
   sequence type, plane, fat suppression and matrix size, which makes it ideal
   for *selecting* which series to feed the model. It must never be a model
   input: it is a site fingerprint, and a model given it will learn the
   institution rather than the pathology. The same caution applies to
   Manufacturer, model and field strength.
8. **Cross-check the provided series flags.** `Fluid_Sensitive`,
   `Fat_Suppression` and `Anatomical_Plane` can be validated against
   `SeriesDescription` tokens (`fs` / `SPAIR` / `we`; `sag` / `cor` / `tra`).
   Quantify disagreement before trusting either source.
9. **Lazy, per-exam DICOM access.** No function may glob the whole tree or
   build an index requiring every file to be read.
