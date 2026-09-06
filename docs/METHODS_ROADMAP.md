# Methods Roadmap

The analytical and numerical methods planned for this project, organized by
where they sit in the pipeline, with the reason each one is appropriate to the
problem rather than merely available.

Written after the exploratory phase, so every motivating number below is
measured. See [`EDA_FINDINGS.md`](EDA_FINDINGS.md) for provenance.

---

## 1. Inference on a 58 examination validation set

**The situation.** The only ground truth is 58 examinations. Positive counts per
label range from 9 (MCL) to 35 (Effusion). Every claim about how well
report-derived labels agree with expert labels rests on this set.

**Why naive point estimates fail here.** An AUC computed from 9 positives and 49
negatives has a standard error large enough that differences of 0.10 between
labels are not distinguishable. Reporting a bare ranking of twelve AUCs invites
weeks of effort optimizing noise.

**Planned methods.**

- **Interval estimation for proportions.** Wilson score intervals rather than
  the normal approximation, which is unreliable at these sample sizes and can
  produce bounds outside [0, 1] when a proportion is near an endpoint.
- **Bootstrap intervals for AUC.** 1000 seeded resamples per label, stratified
  by outcome so no resample loses the minority class entirely. Already
  implemented in `agreement.py`.
- **Beta-binomial posterior for per-label extraction accuracy.** With a weakly
  informative prior, the posterior over accuracy is available in closed form for
  a single proportion, and by MCMC once labels are modeled jointly with a shared
  hyperprior. Partial pooling across the twelve labels is the right structure
  here: a label with 9 positives should borrow strength from the others rather
  than be estimated in isolation.
- **Multiple comparison awareness.** Twelve labels tested simultaneously. Any
  claim that one label extracts better than another needs an adjustment, or an
  explicit statement that the comparison is exploratory.

**Deliverable.** An agreement table where every estimate carries an interval,
and a written statement of which per-label differences are and are not
resolvable at this sample size.

---

## 2. Calibration as a regression diagnostic

**The situation.** Report-derived labels will be systematically biased, not just
noisy. If reports call mild effusion "effusion" while the expert panel requires
moderate, derived positives exceed true positives consistently. The hidden test
set carries expert labels, so a consistent offset transfers straight into the
score.

**Why this is a regression problem.** Calibration asks whether the observed
outcome rate matches the predicted probability across the probability range.
That is a regression of a binary outcome on a predicted value, and it is
assessed exactly as any fit is assessed: by looking at the residual structure.

**Planned methods.**

- **Calibration curves** with the observed rate against predicted probability,
  binned and also smoothed, per label.
- **Logistic recalibration.** Fit `logit(observed) ~ a + b * logit(predicted)`.
  An intercept away from 0 is systematic bias; a slope away from 1 is
  over-confidence or under-confidence. The fitted coefficients give a principled
  correction rather than a hand-tuned threshold shift.
- **Residual inspection by subgroup.** Residuals grouped by report language and
  by scanner manufacturer. Structure in those residuals means the extraction is
  failing differently across sites, which is a different problem from being
  uniformly noisy and needs a different fix.

---

## 3. A tabular baseline before any deep model

**The situation.** There is currently no model of any kind. Jumping straight to
a convolutional network makes it impossible to say later whether the network
earned its complexity.

**Why regression is the right first model.** Several of the twelve findings have
a direct physical signature in image intensity. Joint effusion is fluid, and
fluid is bright on fluid-sensitive sequences. That is a hypothesis a linear
model can test on a handful of interpretable features, and if it fails, the
failure is informative.

**Planned features.** Computed per canonical slot from the normalized volume,
using no scanner or site metadata:

- Intensity percentiles (5th, 25th, 50th, 75th, 95th, 99th)
- Mean, standard deviation, skewness
- Fraction of voxels above the 95th percentile of the same volume, a crude
  proxy for bright fluid extent
- Slot presence indicators, since missingness is itself informative

**Planned methods.**

- **Multiple logistic regression**, one model per label, on the feature set.
- **Transformations.** MRI has no absolute intensity scale, so raw values are
  not comparable across machines. Percentile-based and log transforms are
  candidates, chosen by fit and residual behavior rather than by default.
- **Variable selection.** Stepwise and penalized (LASSO) selection compared, with
  the honest caveat that selection performed inside the training folds must stay
  inside them or the cross-validated estimate is optimistic.
- **Diagnostics.** Influence measures and leverage to find examinations that
  dominate a fit. With 4,349 training rows and heavily imbalanced labels, a
  handful of points can carry a coefficient.

**What a good result looks like.** Not high accuracy. A baseline that beats
chance on Effusion and fails on Fracture would already confirm that the
preprocessing preserves the signal it should and that the label derivation is
not pure noise.

---

## 4. Nonlinear diffusion for denoising

**The situation.** Fluid-sensitive fat-suppressed sequences are noisy, and the
corpus spans 1.5 T and 3.0 T scanners whose noise characteristics differ. Naive
Gaussian smoothing reduces noise but blurs exactly the edges that define a
meniscal tear or a cartilage surface.

**Planned method.** Anisotropic diffusion (Perona-Malik) as an optional
preprocessing stage. The image evolves under

```
    du/dt = div( g(|grad u|) * grad u ),    g(s) = 1 / (1 + (s/K)^2)
```

a nonlinear partial differential equation whose conductance term suppresses
smoothing where the gradient is large. Diffusion proceeds within homogeneous
tissue and stalls at boundaries, which is the behavior this problem wants.

**Numerical treatment.** Explicit finite differences on the voxel grid. The
explicit scheme is conditionally stable, so the time step must respect the
stability bound for the diffusion operator; exceeding it produces oscillation
rather than smoothing. Step count and the conductance parameter K are the two
tunables, and both belong in `PreprocessConfig` so the cache key changes when
they change.

**How it will be judged.** By whether it improves the tabular baseline in
section 3. A preprocessing step that does not move a downstream number is not
kept, however principled it looks.

---

## 5. Label dependence

**The situation.** The twelve labels are not independent. Among the 58, Effusion
and Synovitis are positive together in 22 examinations, and the two meniscus
labels co-occur in 12. Modeling each label with a separate independent
classifier discards that structure.

**Planned methods.**

- **Measured co-occurrence and conditional rates** first, before any modeling
  decision, since with 58 examinations some apparent dependence will be noise.
- **Shared-representation multi-output modeling**, where a common feature set
  feeds twelve outputs, so correlated labels can share evidence.
- **Classifier chains** as a comparison, where a label's prediction may condition
  on others, evaluated for whether the added dependence helps or merely
  propagates error.

---

## 6. Rare label augmentation, stretch goal

**The situation.** MCL has 9 positives among the labeled examinations and the
lowest expected prevalence overall. Any model will be weakest there.

**Planned direction.** Score-based generative modeling, in which sample
generation is the numerical solution of a reverse-time stochastic differential
equation, to synthesize additional positive examples for the rarest findings.

**Stated honestly.** This is speculative, expensive, and carries a real risk of
teaching the model to recognize synthetic artifacts rather than pathology. It is
recorded here as a direction, not a commitment, and it will not be attempted
before sections 1 through 4 are done.

---

## 7. Evaluation protocol

Fixed before modeling begins, so that later choices cannot be tuned to it.

- Five-fold cross-validation grouped by the acquisition proxy, giving folds of
  869, 865, 870, 880 and 865 examinations with no group crossing a fold.
- The 58 expert examinations sit outside all training folds and are used only to
  evaluate label derivation and final calibration.
- Per-label AUC with bootstrap intervals, plus macro AUC across the twelve.
- Any preprocessing or feature selection step is fitted inside the training
  folds only.
- The efficiency leaderboard means model size and preprocessing cost are
  reported alongside accuracy, not after it.

---

## 8. Deliberately not planned

- Rule-based extraction of labels from report text. Section 8.1 of
  `EDA_FINDINGS.md` establishes that the labels are image-derived and that no
  text rule can reproduce them.
- Any use of scanner manufacturer, model, field strength, station name or
  series description as a model feature. All are site fingerprints.
- Threshold tuning against the 58 examinations reported as an accuracy estimate.
  A threshold chosen and evaluated on the same 58 rows is an upper bound, not a
  result, and is labeled as such in `agreement.py`.
