"""Training utility tests. No torch, no competition data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.training import (
    AugmentationConfig,
    TrainingConfig,
    apply_slot_dropout,
    confidence_weights,
    cosine_schedule_with_warmup,
    group_balanced_sample_weights,
    masked_weighted_bce,
    positive_weights,
    soft_target_floor,
)

LABELS = list(config.TARGET_LABELS)


def _frame(values):
    return pd.DataFrame({label: values for label in LABELS})


def test_positive_weight_is_the_negative_to_positive_ratio():
    labels = _frame([1.0] * 5 + [0.0] * 20)
    weights = positive_weights(labels)
    assert weights["ACL"] == pytest.approx(4.0)


def test_positive_weight_is_clipped():
    labels = _frame([1.0] * 5 + [0.0] * 500)
    assert positive_weights(labels, clip=10.0)["ACL"] == pytest.approx(10.0)


def test_a_label_with_too_few_positives_is_not_upweighted():
    labels = _frame([1.0] * 2 + [0.0] * 100)
    assert positive_weights(labels, minimum_positives=5)["ACL"] == pytest.approx(1.0)


def test_confidence_rises_with_distance_from_one_half():
    frame = _frame([0.5, 0.75, 1.0])
    weights = confidence_weights(frame)
    column = weights["ACL"].tolist()
    assert column[0] < column[1] < column[2]


def test_confidence_is_floored_not_zeroed():
    weights = confidence_weights(_frame([0.5]), floor=0.1)
    assert weights["ACL"].iloc[0] == pytest.approx(0.1)


def test_a_chance_level_label_contributes_only_the_floor():
    frame = _frame([1.0])
    weights = confidence_weights(frame, agreement={label: 0.5 for label in LABELS}, floor=0.1)
    assert weights["ACL"].iloc[0] == pytest.approx(0.1)


def test_targets_are_kept_away_from_the_endpoints():
    floored = soft_target_floor(_frame([0.0, 1.0]), epsilon=0.02)
    assert floored["ACL"].tolist() == [pytest.approx(0.02), pytest.approx(0.98)]


def test_group_weights_favor_the_small_group():
    groups = ["big"] * 90 + ["small"] * 10
    weights = group_balanced_sample_weights(groups, temperature=1.0)
    assert weights[0] < weights[-1]


def test_zero_temperature_is_uniform_sampling():
    weights = group_balanced_sample_weights(["big"] * 90 + ["small"] * 10, temperature=0.0)
    assert np.allclose(weights, 1.0)


def test_horizontal_flip_is_refused():
    with pytest.raises(ValueError, match="medial and lateral"):
        AugmentationConfig(horizontal_flip=True)


def test_slot_dropout_never_empties_an_exam():
    rng = np.random.default_rng(0)
    mask = np.array([True, True, True, True])
    for _ in range(200):
        assert apply_slot_dropout(mask, dropout=0.99, rng=rng).any()


def test_slot_dropout_leaves_a_single_viewpoint_alone():
    rng = np.random.default_rng(0)
    mask = np.array([True, False, False, False])
    assert apply_slot_dropout(mask, dropout=0.99, rng=rng).tolist() == mask.tolist()


def test_slot_dropout_never_revives_an_absent_viewpoint():
    rng = np.random.default_rng(1)
    mask = np.array([True, True, False, False])
    for _ in range(50):
        result = apply_slot_dropout(mask, dropout=0.5, rng=rng)
        assert not result[2] and not result[3]


def test_schedule_warms_up_then_decays():
    total = 100
    assert cosine_schedule_with_warmup(0, total) < 1.0
    assert cosine_schedule_with_warmup(10, total) == pytest.approx(1.0, abs=0.05)
    assert cosine_schedule_with_warmup(99, total) < 0.05


def test_encoder_must_learn_slower_than_the_head():
    with pytest.raises(ValueError, match="encoder"):
        TrainingConfig(learning_rate=1e-5, encoder_learning_rate=1e-3)


def test_unobserved_labels_contribute_nothing_to_the_loss():
    predictions = np.array([[0.9, 0.1]])
    targets = np.array([[1.0, 1.0]])
    both = masked_weighted_bce(predictions, targets, np.array([[True, True]]))
    first_only = masked_weighted_bce(predictions, targets, np.array([[True, False]]))
    assert first_only < both


def test_loss_is_zero_when_nothing_is_observed():
    assert masked_weighted_bce(
        np.array([[0.5]]), np.array([[1.0]]), np.array([[False]])
    ) == 0.0


def test_a_confident_correct_prediction_loses_less_than_a_wrong_one():
    observed = np.array([[True]])
    good = masked_weighted_bce(np.array([[0.95]]), np.array([[1.0]]), observed)
    bad = masked_weighted_bce(np.array([[0.05]]), np.array([[1.0]]), observed)
    assert good < bad
