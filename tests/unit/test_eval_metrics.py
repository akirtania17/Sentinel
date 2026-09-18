"""Metrics against small datasets computed by hand."""

import pytest

from sentinel_runtime import ChoiceResult, NoulResult, ScoreResult
from sentinel_runtime.evals.fixtures import ScoreRange
from sentinel_runtime.evals.metrics import (
    SWEEP_THRESHOLDS,
    Observation,
    average_precision,
    brier,
    calibration,
    choice_metrics,
    confusion,
    noul_metrics,
    percentile,
    roc_auc,
    score_metrics,
)

PROBABILITIES = [0.9, 0.8, 0.7, 0.4, 0.3, 0.1]
LABELS = [True, True, False, True, False, False]


def test_confusion_at_threshold() -> None:
    at = confusion(PROBABILITIES, LABELS, 0.5)
    assert (at.tp, at.fp, at.tn, at.fn) == (2, 1, 2, 1)
    assert at.precision == pytest.approx(2 / 3, abs=1e-6)
    assert at.recall == pytest.approx(2 / 3, abs=1e-6)
    assert at.f1 == pytest.approx(2 / 3, abs=1e-6)
    assert at.accuracy == pytest.approx(4 / 6, abs=1e-6)
    assert at.fpr == pytest.approx(1 / 3, abs=1e-6)


def test_threshold_is_inclusive() -> None:
    assert confusion([0.5], [True], 0.5).tp == 1


def test_rank_metrics() -> None:
    assert roc_auc(PROBABILITIES, LABELS) == pytest.approx(8 / 9, abs=1e-6)
    assert average_precision(PROBABILITIES, LABELS) == pytest.approx(1 / 3 + 1 / 3 + 0.25, abs=1e-6)
    assert brier(PROBABILITIES, LABELS) == pytest.approx(1.0 / 6, abs=1e-6)


def test_ties_count_half_in_roc_auc() -> None:
    assert roc_auc([0.5, 0.5], [True, False]) == 0.5


def test_undefined_values_are_none_not_zero() -> None:
    no_predicted = confusion(PROBABILITIES, LABELS, 0.95)
    assert no_predicted.precision is None
    assert no_predicted.f1 is None
    assert no_predicted.recall == 0.0
    only_positive = confusion([0.9, 0.2], [True, True], 0.5)
    assert only_positive.fpr is None
    assert roc_auc([0.9, 0.2], [True, True]) is None
    assert average_precision([0.9, 0.2], [False, False]) is None
    assert brier([], []) is None
    assert confusion([], [], 0.5).accuracy is None


def test_calibration_bins_and_ece() -> None:
    bins, ece = calibration(PROBABILITIES, LABELS)
    assert len(bins) == 10
    assert ece is None, "ECE needs at least 10 labeled cases"
    probabilities = [0.05] * 10 + [0.95] * 10
    labels = [False] * 10 + [True] * 10
    bins, ece = calibration(probabilities, labels)
    assert ece == pytest.approx(0.05, abs=1e-6)
    assert bins[0].n == 10
    assert bins[9].n == 10
    assert bins[9].positive_rate == 1.0
    _, edge = calibration([1.0] * 10, [True] * 10)
    assert edge == 0.0


def _noul_observations() -> list[Observation]:
    observations = [
        Observation(case_id=f"c{i}", result=NoulResult(probability=p), expected=label, labeled=True)
        for i, (p, label) in enumerate(zip(PROBABILITIES, LABELS, strict=True))
    ]
    observations.append(
        Observation(case_id="amb1", result=NoulResult(probability=0.5), expected=None, labeled=True)
    )
    observations.append(
        Observation(case_id="amb2", result=NoulResult(probability=0.02), expected=None, labeled=True)
    )
    observations.append(
        Observation(case_id="free", result=NoulResult(probability=0.9), expected=None, labeled=False)
    )
    return observations


def test_noul_metrics() -> None:
    metrics = noul_metrics(_noul_observations(), threshold=0.5, threshold_source="default", max_fpr=0.34)
    assert metrics.evaluated == 9
    assert metrics.labeled == 6
    assert (metrics.positives, metrics.negatives, metrics.ambiguous) == (3, 3, 2)
    assert metrics.headline() == ("f1", metrics.at_threshold.f1)
    assert [row.threshold for row in metrics.sweep] == list(SWEEP_THRESHOLDS)
    assert metrics.fragile == ["c3"], "0.4 is within 0.10 of 0.5"
    assert metrics.ambiguous_in_band == 1
    assert metrics.ambiguous_confident == ["amb2"]
    assert metrics.ambiguous_mean == pytest.approx(0.26)
    best_f1 = max(row.f1 for row in metrics.sweep if row.f1 is not None)
    assert metrics.best_f1.value == best_f1
    assert metrics.best_recall_at_fpr.fpr is not None
    assert metrics.best_recall_at_fpr.fpr <= 0.34
    assert any("ECE" in note for note in metrics.notes)


def test_noul_metrics_with_one_class() -> None:
    observations = [Observation(case_id="a", result=NoulResult(probability=0.9), expected=True, labeled=True)]
    metrics = noul_metrics(observations, threshold=0.5, threshold_source="predicate", max_fpr=0.05)
    assert metrics.roc_auc is None
    assert metrics.best_recall_at_fpr.threshold is None
    assert any("only one class" in note for note in metrics.notes)


def _choice(choice: str, confidence: float, probabilities: dict[str, float] | None = None) -> ChoiceResult:
    distribution = probabilities or {option: (1.0 if option == choice else 0.0) for option in ("a", "b", "c")}
    return ChoiceResult(choice=choice, probabilities=distribution, confidence=confidence)


def test_choice_metrics() -> None:
    observations = [
        Observation(case_id="1", result=_choice("a", 0.9), expected="a", labeled=True),
        Observation(case_id="2", result=_choice("b", 0.4), expected="a", labeled=True),
        Observation(case_id="3", result=_choice("b", 0.6), expected=["b", "c"], labeled=True),
        Observation(case_id="4", result=_choice("c", 0.95), expected="c", labeled=True),
        Observation(case_id="5", result=_choice("a", 0.3), expected=None, labeled=True),
    ]
    metrics = choice_metrics(observations, options=["a", "b", "c"], min_confidence=0.5)
    assert metrics.labeled == 4
    assert metrics.ambiguous == 1
    assert metrics.accuracy == 0.75
    assert metrics.confusion == {
        "a": {"a": 1, "b": 1, "c": 0},
        "b": {"a": 0, "b": 0, "c": 0},
        "c": {"a": 0, "b": 0, "c": 1},
    }
    assert metrics.options["a"].precision == 1.0
    assert metrics.options["a"].recall == 0.5
    assert metrics.options["b"].precision == 0.0
    assert metrics.options["b"].recall is None
    buckets = {(b.lower, b.upper): b for b in metrics.by_confidence}
    assert buckets[(0.0, 0.5)].accuracy == 0.0
    assert buckets[(0.7, 0.9)].accuracy is None
    assert buckets[(0.9, 1.0)].n == 2
    assert metrics.at_min_confidence is not None
    assert metrics.at_min_confidence.coverage == 0.75
    assert metrics.at_min_confidence.accuracy == 1.0
    assert any("several options" in note for note in metrics.notes)


def test_score_metrics() -> None:
    def result(score: float, level: int) -> ScoreResult:
        return ScoreResult(
            score=score,
            probabilities={0: 1.0},
            confidence=0.5,
            levels=5,
            normalized=score / 4,
            nearest_level=level,
        )

    observations = [
        Observation(case_id="1", result=result(1.2, 1), expected=1, labeled=True),
        Observation(case_id="2", result=result(2.6, 3), expected=2, labeled=True),
        Observation(case_id="3", result=result(1.9, 2), expected=ScoreRange(min=2), labeled=True),
        Observation(case_id="4", result=result(0.4, 0), expected=ScoreRange(min=1, max=2), labeled=True),
    ]
    metrics = score_metrics(observations)
    assert (metrics.exact_labeled, metrics.range_labeled) == (2, 2)
    assert metrics.exact_accuracy == 0.5
    assert metrics.within_one == 1.0
    assert metrics.mae == pytest.approx(0.4)
    assert metrics.range_satisfaction == 0.5
    assert metrics.range_mae == pytest.approx(0.05), "1.9 rounds into {min: 2}; 0.4 is 0.1 below 0.5"
    assert metrics.headline() == ("exact_accuracy", 0.5)
    ranges_only = score_metrics(observations[2:])
    assert ranges_only.headline() == ("range_satisfaction", 0.5)


def test_score_range_helpers() -> None:
    band = ScoreRange(min=1, max=2)
    assert band.contains(1)
    assert not band.contains(3)
    assert band.distance(2.9) == pytest.approx(0.4)
    assert band.distance(0.2) == pytest.approx(0.3)
    assert band.distance(1.6) == 0.0
    assert ScoreRange(max=1).distance(0.2) == 0.0
    with pytest.raises(ValueError, match="needs `min`, `max`"):
        ScoreRange()
    with pytest.raises(ValueError, match="above max"):
        ScoreRange(min=3, max=1)


def test_percentile() -> None:
    assert percentile([], 0.5) is None
    assert percentile([5.0, 1.0, 3.0], 0.5) == 3.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.95) == 4.0
