# Label extraction prompt v1

You are producing **soft supervision**, not ground truth. The twelve labels below
were assigned from MRI images. The radiology report is an imperfect proxy and may
contradict the image-derived label. Do not force a confident binary decision when
the report is ambiguous, incomplete, internally inconsistent, or only suggests a
finding.

Read the complete report as one document. Do not assume that section headers,
language, punctuation, spelling, or formatting are reliable. The report may be in
English, Spanish, Turkish, Croatian, Greek, German, Bulgarian, Dutch, or French.

For each label, return a probability in [0, 1]. The probability should represent
how strongly the report supports the image-derived abnormality, not how strongly
a keyword appears. Use values near 0.5 when the text is uncertain or contradictory.

Labels, exactly as spelled:
- ACL
- MCL
- Medial Meniscus
- Lateral Meniscus
- Medial OA
- Lateral OA
- PF OA
- Effusion
- Synovitis
- Baker's
- Contusion
- Fracture

Return exactly one JSON object with this shape:

~~~json
{
  "probabilities": {
    "ACL": 0.0,
    "MCL": 0.0,
    "Medial Meniscus": 0.0,
    "Lateral Meniscus": 0.0,
    "Medial OA": 0.0,
    "Lateral OA": 0.0,
    "PF OA": 0.0,
    "Effusion": 0.0,
    "Synovitis": 0.0,
    "Baker's": 0.0,
    "Contusion": 0.0,
    "Fracture": 0.0
  },
  "confidence": {
    "ACL": 0.0,
    "MCL": 0.0,
    "Medial Meniscus": 0.0,
    "Lateral Meniscus": 0.0,
    "Medial OA": 0.0,
    "Lateral OA": 0.0,
    "PF OA": 0.0,
    "Effusion": 0.0,
    "Synovitis": 0.0,
    "Baker's": 0.0,
    "Contusion": 0.0,
    "Fracture": 0.0
  },
  "rationale": "One sentence naming the strongest evidence and quoting a short report phrase."
}
~~~

The confidence values are your confidence in the probability assignment, also in
[0, 1]. The rationale must be one sentence and must quote a short phrase from the
report. Do not mention or infer any label from information outside the report.

## Ambiguity guidance from validation

The 58 expert-labeled exams show that report language is not a deterministic
label definition. The observed extractability ranking is:

- Relatively more extractable: PF OA, ACL, MCL, Baker's.
- Intermediate and inconsistent: Medial Meniscus, Lateral Meniscus, Medial OA,
  Lateral OA, Synovitis.
- Least reliable from text: Effusion, Contusion, Fracture.

These are validation observations, not permission to use keyword rules. In
particular, similar findings can receive opposite image-derived labels.

Negation and uncertainty are multilingual and must be interpreted in context,
not matched as isolated tokens. Examples of surface forms include:

- English: "intact", "normal", "preserved", "without", "no evidence of",
  "not torn", "suspect", "R/O", "likely", "possible".
- Spanish: "sin", "no hay", "sin signos de", "sin alteraciones",
  "no impresiona", "probablemente", "podría".
- Turkish: "normal", "korunmuş", "uyumlu", "düşündüren", and grade or
  rupture wording that may be tentative.
- Greek: "δεν παρατηρούνται", "φυσιολογικά", "χωρίς", "πιθανής",
  "θα μπορούσε να αποδοθεί".
- Croatian: "bez znakova", "održanog kontinuiteta", "moguće", "vjerojatno".
- Bulgarian: "без данни за", "запазен", "нормално представени".
- Dutch: "normaal", "zonder", "geen", "mogelijk", "waarschijnlijk".
- German: "intakt", "ohne", "kein", "nicht sicher", "wohl",
  "möglicherweise".

The token "intact" may be corrupted when immediately followed by a digit in the
source data, for example "intact9xintact4cm" represents a measurement. Do not
interpret that form as an anatomic negation.

Report:

{report}
