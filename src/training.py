"""Training utilities built for noisy labels, missing inputs, and site shift.

Nothing here assumes the data is clean. Labels arrive as probabilities derived
from radiology reports that disagree with the expert panel at a rate we measure
rather than assume; inputs arrive with viewpoints missing for roughly one exam
in twenty; and the corpus spans around nineteen institutions whose scans differ
systematically. Each of those is handled explicitly below rather than left to
the optimizer.

The torch dependency is imported lazily inside the functions that need it, so
the weighting, sampling and schedule logic stays importable and testable on a
machine with no deep learning stack installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# Class imbalance
# ---------------------------------------------------------------------------

def positive_weights(
    labels: pd.DataFrame,
    clip: float = 10.0,
    minimum_positives: int = 5,
) -> dict[str, float]:
    """Per label weight on the positive term of a binary cross entropy loss.

    The twelve findings range from roughly 16 percent prevalence to 60 percent,
    so a single unweighted loss lets the common findings dominate the gradient
    and the rare ones collapse to predicting the majority class. The weight is
    the ratio of negatives to positives, which equalizes the expected
    contribution of each class.

    The ratio is clipped because an unclipped weight on a very rare label makes
    training unstable: a handful of positives then carry most of the gradient
    and the loss oscillates. A label with fewer than minimum_positives
    observations gets a weight of 1.0 and should be treated as unlearnable
    rather than aggressively upweighted.
    """
    weights: dict[str, float] = {}
    for label in config.TARGET_LABELS:
        if label not in labels.columns:
            weights[label] = 1.0
            continue
        column = labels[label].dropna()
        positives = float((column >= 0.5).sum())
        negatives = float((column < 0.5).sum())
        if positives < minimum_positives or negatives <= 0:
            weights[label] = 1.0
        else:
            weights[label] = float(min(negatives / positives, clip))
    return weights


# ---------------------------------------------------------------------------
# Label noise
# ---------------------------------------------------------------------------

def confidence_weights(
    soft_labels: pd.DataFrame,
    agreement: Mapping[str, float] | None = None,
    floor: float = 0.1,
) -> pd.DataFrame:
    """Per example, per label weight reflecting how much to trust a soft label.

    Two independent sources of doubt are combined.

    The first is the extractor's own uncertainty on this example. A probability
    near 0.5 means the report did not settle the question, and training hard on
    it teaches the model to reproduce the extractor's confusion. Weight scales
    with distance from 0.5.

    The second is how well that label extracts at all, measured against the 58
    expert examinations. A label whose derived values agree with the panel at
    chance should contribute almost nothing regardless of how confident the
    extractor sounded on any single report.

    A floor keeps every example contributing something, since zeroing rows
    silently shrinks the training set in a way that is easy to miss.
    """
    columns = [label for label in config.TARGET_LABELS if label in soft_labels.columns]
    values = soft_labels.loc[:, columns].astype(float)

    # Distance from maximal uncertainty, rescaled to [0, 1].
    certainty = (values - 0.5).abs() * 2.0

    if agreement is not None:
        # Map AUC to a usable multiplier: 0.5 is chance and contributes nothing
        # beyond the floor, 1.0 contributes fully.
        scale = pd.Series(
            {label: max(0.0, (float(agreement.get(label, 0.5)) - 0.5) * 2.0) for label in columns}
        )
        certainty = certainty.mul(scale, axis=1)

    return certainty.clip(lower=floor, upper=1.0).fillna(floor)


def soft_target_floor(values: pd.DataFrame, epsilon: float = 0.02) -> pd.DataFrame:
    """Keep targets away from exactly 0 and 1.

    Derived labels are not certain, and a target of exactly 1.0 asks the model
    to be infinitely confident about something a language model inferred from a
    sentence. Clamping is the same mechanism as label smoothing, applied here
    for a concrete reason rather than as a default.
    """
    if not 0.0 <= epsilon < 0.5:
        raise ValueError("epsilon must lie in [0, 0.5).")
    return values.clip(lower=epsilon, upper=1.0 - epsilon)


# ---------------------------------------------------------------------------
# Site shift
# ---------------------------------------------------------------------------

def group_balanced_sample_weights(
    groups: Sequence[str] | pd.Series,
    temperature: float = 0.5,
) -> np.ndarray:
    """Sampling weights that stop the largest site from dominating a batch.

    The largest acquisition group holds 590 examinations and the smallest holds
    13. Sampling uniformly means the model sees the large group roughly forty
    five times as often and tunes itself to that site's protocol.

    Weighting by the inverse group size corrects this completely, but then the
    thirteen examination group is oversampled hard enough to be memorized.
    Temperature interpolates: 0 is uniform sampling, 1 is full inverse
    frequency, and 0.5 is a reasonable compromise.
    """
    if not 0.0 <= temperature <= 1.0:
        raise ValueError("temperature must lie in [0, 1].")
    series = pd.Series(list(groups))
    counts = series.map(series.value_counts())
    weights = counts.to_numpy(dtype=float) ** (-float(temperature))
    return weights / weights.mean()


@dataclass(frozen=True)
class AugmentationConfig:
    """Augmentations chosen for the distortions this corpus actually contains.

    Horizontal flip is deliberately excluded. Left and right knees are mirror
    images, and medial versus lateral is the difference between six of the
    twelve labels, so a horizontal flip relabels the exam.
    """

    # Scanner to scanner intensity differences are the dominant nuisance
    # variable, so brightness and contrast jitter are the primary augmentation.
    brightness: float = 0.15
    contrast: float = 0.15
    gamma: tuple[float, float] = (0.8, 1.25)

    # Small geometric jitter for positioning differences between sites.
    rotation_degrees: float = 8.0
    translate_fraction: float = 0.06
    scale: tuple[float, float] = (0.92, 1.08)

    # Mild noise, since field strength differs across the corpus and 1.5 T
    # images are noisier than 3.0 T.
    gaussian_noise_sigma: float = 0.02

    # Randomly drop a present viewpoint during training so the model cannot
    # assume all four are available at inference. Roughly one exam in twenty is
    # genuinely missing a viewpoint, so the model must degrade gracefully.
    slot_dropout: float = 0.15

    horizontal_flip: bool = False

    def __post_init__(self) -> None:
        if self.horizontal_flip:
            raise ValueError(
                "Horizontal flip swaps medial and lateral, which relabels six of "
                "the twelve findings. It must stay disabled."
            )
        if not 0.0 <= self.slot_dropout < 1.0:
            raise ValueError("slot_dropout must lie in [0, 1).")


# ---------------------------------------------------------------------------
# Missing inputs
# ---------------------------------------------------------------------------

def apply_slot_dropout(
    presence_mask: np.ndarray,
    dropout: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Randomly mark present viewpoints as absent, never emptying an exam.

    Dropping every viewpoint would hand the model an input with no signal and a
    target that still demands twelve predictions, which is noise rather than
    regularization. At least one present viewpoint always survives.
    """
    mask = np.asarray(presence_mask, dtype=bool).copy()
    present = np.flatnonzero(mask)
    if len(present) <= 1 or dropout <= 0.0:
        return mask
    keep = rng.random(len(present)) >= dropout
    if not keep.any():
        keep[rng.integers(len(present))] = True
    mask[present] = keep
    return mask


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainingConfig:
    """Settings for one fold. Defaults chosen for a single mid range GPU."""

    epochs: int = 12
    batch_size: int = 8
    learning_rate: float = 3e-4
    encoder_learning_rate: float = 3e-5
    weight_decay: float = 1e-4
    warmup_fraction: float = 0.1
    gradient_clip: float = 1.0
    early_stopping_patience: int = 3
    mixed_precision: bool = True
    seed: int = config.RANDOM_SEED
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    group_sampling_temperature: float = 0.5
    label_epsilon: float = 0.02
    positive_weight_clip: float = 10.0

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive.")
        if not 0.0 <= self.warmup_fraction < 1.0:
            raise ValueError("warmup_fraction must lie in [0, 1).")
        if self.encoder_learning_rate > self.learning_rate:
            raise ValueError(
                "The pretrained encoder should move slower than the new head; "
                "set encoder_learning_rate below learning_rate."
            )


def cosine_schedule_with_warmup(
    step: int, total_steps: int, warmup_fraction: float = 0.1
) -> float:
    """Learning rate multiplier for a given step.

    Warmup matters here because the head is randomly initialized while the
    encoder is pretrained. Without it the first large gradients from the random
    head propagate back and damage the pretrained features.
    """
    if total_steps < 1:
        raise ValueError("total_steps must be positive.")
    warmup_steps = max(1, int(total_steps * warmup_fraction))
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))


def masked_weighted_bce(
    predictions: np.ndarray,
    targets: np.ndarray,
    observed: np.ndarray,
    example_weights: np.ndarray | None = None,
    positive_weight: np.ndarray | None = None,
    epsilon: float = 1e-7,
) -> float:
    """Reference implementation of the training loss, in NumPy.

    Kept framework free so the loss can be checked by hand and unit tested. The
    torch version in the training loop must agree with this one.

    Unobserved labels contribute nothing. That is the whole point: 4,349 of the
    examinations have no expert label, and treating a missing label as a
    negative would teach the model that every unlabeled knee is healthy.
    """
    predictions = np.clip(np.asarray(predictions, dtype=float), epsilon, 1.0 - epsilon)
    targets = np.asarray(targets, dtype=float)
    observed = np.asarray(observed, dtype=bool)

    positive_term = targets * np.log(predictions)
    negative_term = (1.0 - targets) * np.log(1.0 - predictions)
    if positive_weight is not None:
        positive_term = positive_term * np.asarray(positive_weight, dtype=float)
    loss = -(positive_term + negative_term)

    weights = observed.astype(float)
    if example_weights is not None:
        weights = weights * np.asarray(example_weights, dtype=float)

    total = weights.sum()
    if total <= 0:
        return 0.0
    return float((loss * weights).sum() / total)
