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

The 58 labeled exams span **eight** of the nine languages — en 28, es 10, tr 6,
hr 4, bg 3, el 3, nl 2, de 2. **French has zero labeled exams**, so French
extraction can never be validated. With only 2–4 exams for de/nl/el/bg,
extractor accuracy can be validated **only in aggregate**, never per language.

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

### 6.5 Grouping key — measured, not aspirational

The five-part key `(normalized_manufacturer, model, MagneticFieldStrength,
detected_language, placeholder_signature)` **does not survive contact with the
corpus.** Measured over all 4,407 exams:

| min_group_size | groups | % keeping full key | smallest | median | largest | largest % | singletons |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 25 | 16.2 | 8 | 137 | 590 | 13.4 | 0 |
| 10 | 25 | 16.0 | 8 | 137 | 590 | 13.4 | 0 |
| 15 | 20 | 8.4 | 8 | 215 | 590 | 13.4 | 0 |
| 20 | 19 | 8.4 | 24 | 220 | 590 | 13.4 | 0 |
| 30 | 19 | 8.4 | 24 | 220 | 590 | 13.4 | 0 |

Even at a threshold of 5, only 16% of exams retain the full key — it fragments
into hundreds of combinations of 1–4 exams. The threshold is not a meaningful
tuning knob; the ceiling is structural.

**The realized grouping is two-part: `(normalized_manufacturer,
detected_language)`.** Roughly 91% of exams coarsen to this level at
min_group_size=30, 84% at min_group_size=10.

**Decision: `min_group_size=10`** — 25 groups, no singletons, smallest group 8.

**Why 5-fold CV is still viable.** The largest group is 590 exams; one fifth of
the corpus is 881. No group is too large to sit inside a single fold, which is
the constraint that would otherwise break grouped splitting.

**What this costs.** Grouping by (manufacturer, language) is *coarser* than
grouping by site — all Spanish Siemens sites collapse together. Coarser grouping
makes CV stricter, not looser, so this errs safe on leakage. The real cost is
fewer groups, which makes fold balance lumpier and CV estimates noisier.

**Note on unmapped vendors.** `normalize_manufacturer` slugified two unrecognized
vendor strings rather than failing — `FUJIFILM Healthcare Corporation` and
`Hitachi Medical Corporation` (24 exams total) passed through as their own
groups and coarsened all the way to language level. Both are one lineage
(Fujifilm acquired Hitachi's diagnostic imaging business) and should map to
`fujifilm`. Unrecognized vendors must warn, never pass through silently.

### 6.6 Verified vendor counts (all 4,407 exams)

```
Siemens Healthineers 1053 | SIEMENS 804 | Siemens 94        -> siemens   1951
Philips Medical Systems 718 | Philips 492 | Philips Hc 91   -> philips   1301
GE MEDICAL SYSTEMS 868 | GEHC 37                            -> ge         905
TOSHIBA 181 | CANON_MEC 45                                  -> canon      226
FUJIFILM Healthcare 16 | Hitachi Medical 8                  -> fujifilm    24
```

Sums to 4,407. Full-corpus DICOM metadata read in 76 seconds with zero
unreadable exams.

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

### 8.1 RESOLVED — the labels do not come from the reports

All 58 labeled exams were read against their labels. Positive-exam counts
reconcile exactly with `train.csv` for all twelve labels, so the analysis below
rests on a correct parse.

**Finding: no text rule can exist, because the labels are image-derived.**

Several exams carry a positive label while the report states the opposite:

| exam | label | report says |
|---|---|---|
| 57 | Baker's = 1 | explicitly no Baker cyst |
| 41 | Lateral Meniscus = 1 | "Normaal voorkomen menisci" |
| 51 | Lateral OA = 1 | "Cartilages normal" |
| 22 | Medial OA = 1 | "Cartilage appears intact" |

No threshold or severity rule can turn a negative statement into a positive
label. The ground truth was assigned by an expert panel **reading the images**.
The reports are independent clinical documents written by different
radiologists at 16–19 institutions under no shared protocol.

The contradictions are therefore report-versus-image disagreement, not
annotation noise. Searching for exact operational rules is wasted effort.

### 8.2 Per-label extractability

Derived from the 58. Treat as noisy guidance, not specification.

**Relatively consistent**

| label | apparent boundary | confidence |
|---|---|---|
| PF OA | moderate/high-grade patellofemoral disease, full-thickness loss, grade 3–4 chondromalacia positive; mild/superficial chondrosis negative | medium-high |
| ACL | complete or high-grade tear positive; grade 1–2 sprain, mucoid degeneration mostly negative | medium |
| MCL | grade II or greater positive; grade I sprain, thickening, edema negative | medium |
| Baker's | explicitly named cyst usually positive; size threshold does not hold | low-medium |

**Noisy weak signal:** Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA,
Synovitis. Tear-versus-degeneration is visible on the medial side but breaks
down laterally; OA compartment labels contain direct reversals.

**No recoverable text rule:** Effusion, Contusion, Fracture.

- Effusion: "Leve derrame articular" (mild) is positive in exam 1 and negative
  in exam 35. "Moderate joint effusion" is negative in exam 2 while "Small joint
  effusion" is positive in exam 9. No defensible threshold explains these.
- Contusion: exam 20 uses "kemik kontüzyonu" verbatim and is negative; exam 6
  says "no fracture or bone contusion" and is positive.
- Fracture: exam 2 has an explicit osteochondral fracture and is negative; exam
  14 has insufficiency fractures and is negative.

### 8.3 Label relationships

- Effusion and Synovitis co-occur positively in 22 exams but each appears
  without the other frequently. Not equivalent.
- Medial and lateral meniscus labels co-occur in 12 exams. Not exclusive.
- The three OA compartment labels are not mutually exclusive.
- No label is a strict prerequisite for another.

### 8.4 Negation and uncertainty inventory

| language | negation | uncertainty |
|---|---|---|
| en | intact, normal, preserved, without, no evidence of, no frank, not torn | suspect, R/O, likely, possible, cannot totally excluded |
| es | sin, no hay, sin signos de, sin alteraciones, no impresiona | probablemente, podría |
| tr | normal, korunmuş, seçilememiştir | uyumlu, düşündüren |
| el | δεν παρατηρούνται, φυσιολογικά, χωρίς | πιθανής |
| hr | bez znakova, održanog kontinuiteta | moguće, vjerojatno, najvjerojatnije |
| bg | без данни за, запазен, нормално | — |
| nl | normaal, zonder, geen | mogelijks |
| de | intakt, ohne, kein, nicht sicher | möglicherweise, wohl |

`intact` is unusable as a negation cue until the section 5 repair has run.

### 8.5 Evaluation metric

Probability-valued submission is consistent with an AUC metric, and a separate
efficiency leaderboard is documented externally. Confirm both on the Evaluation
tab.

### 8.6 Still open

The competition's published labeling guideline has not been read. It may define
the panel's criteria explicitly and is worth finding before further inference.

---

## 8.7 Canonical image-slot coverage (all 4,407 exams)

Measured from train_series.csv using the four metadata-only canonical routing slots. Missing slots are retained as explicit missing inputs rather than causing an exam to be discarded.

| slot | selected | missing | coverage |
|---|---:|---:|---:|
| sagittal fluid-sensitive | 4,150 | 257 | 94.1684% |
| coronal fluid-sensitive | 4,248 | 159 | 96.3921% |
| axial fluid-sensitive | 4,407 | 0 | 100.0000% |
| sagittal T1 | 4,266 | 141 | 96.8005% |

The downstream sample contract must therefore include one presence mask per slot. Axial fluid-sensitive coverage is complete; the remaining slots have measurable gaps and must not be silently imputed or treated as negative evidence.
## 9. Design implications

1. **Image-only at inference.** Reports are training-time supervision. Nothing
   in the inference path may touch text.
2. **Do not build a rule-based extractor.** Section 8.1 establishes that no
   deterministic text rule can reproduce image-derived labels. Effort spent on
   keyword and negation rules is capped well below useful accuracy.
3. **Label generation is offline preprocessing, not part of the submission.**
   The internet-off constraint applies only to the inference notebook. Labels
   for the 4,349 unlabeled reports can be generated once by any strong LLM, run
   anywhere, and saved as a Kaggle Dataset attached to training. Verify the
   competition rules on external pretrained models before relying on this.
4. **Ask for soft labels, not binaries.** A per-label probability carries
   extraction uncertainty into training, which is the correct shape for noisy
   supervision.
5. **Measure per-label agreement against the 58.** That number estimates the
   noise injected per label and identifies which labels are worth training on.
6. **Systematic bias is worse than random noise.** If reports call mild effusion
   "effusion" while the panel requires moderate-plus, derived labels are
   systematically over-positive and the model inherits that shift against an
   expert-labeled test set. Check per-label positive rates of derived labels
   against the 58 and correct systematic offsets.
7. **NaN is unknown, never zero.** Label parsing returns an observed-mask
   alongside values. Counts reported as positive / negative / missing.
8. **French has no labeled exams** (81 in the corpus). Any French extraction is
   entirely unvalidated — state this as a limitation.
9. **Reports contain ordinary dictation typos** ("less thab 50%") in all nine
   languages. Exact-match extraction is brittle.
10. **Group CV** by the section 6.5 key, never randomly.
11. **`SeriesDescription` is for routing, not features.** It reliably encodes
    sequence type, plane, fat suppression and matrix size, making it ideal for
    *selecting* which series to feed the model. It must never be a model input:
    it is a site fingerprint, and a model given it will learn the institution
    rather than the pathology. Same caution for Manufacturer, model and field
    strength.
12. **Cross-check the provided series flags** against `SeriesDescription`
    tokens (`fs` / `SPAIR` / `we`; `sag` / `cor` / `tra`) and quantify
    disagreement before trusting either source.
13. **Lazy, per-exam DICOM access.** No function may glob the whole tree or
    build an index requiring every file to be read.
14. **Never commit report text.** This repository is public and Kaggle
    prohibits redistributing competition data.
