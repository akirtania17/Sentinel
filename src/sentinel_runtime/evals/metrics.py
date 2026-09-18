"""Predicate quality metrics.

All functions here are pure: they take observations (a result and its label) and return metrics.
Undefined values are ``None``, never 0: precision with no predicted positives, recall with no
labeled positives, ROC-AUC or PR-AUC with a single labeled class, ECE with fewer than 10 labeled
cases. A note explains each undefined value.

Nouls are scored at an operating threshold and swept from 0.05 to 0.95. Choices are scored by
accuracy (a list of acceptable options counts as correct when the choice is in it), per option, by
confidence bucket, and by abstention threshold. Scores are scored at the nearest level, within one
level, and against ranges.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from sentinel_runtime.evals.fixtures import Expectation, ScoreRange
from sentinel_runtime.results import ChoiceResult, NoulResult, Result, ScoreResult

SWEEP_THRESHOLDS: tuple[float, ...] = tuple(round(0.05 * step, 2) for step in range(1, 20))
ABSTENTION_THRESHOLDS: tuple[float, ...] = tuple(round(0.5 + 0.05 * step, 2) for step in range(10))
CONFIDENCE_BUCKETS: tuple[tuple[float, float], ...] = ((0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0))
FRAGILITY_MARGIN = 0.10
ECE_BINS = 10
MIN_ECE_CASES = 10
AMBIGUOUS_BAND = (0.3, 0.7)
CONFIDENT = 0.1
"""Nouls below this or above ``1 - CONFIDENT`` on an ambiguous case are flagged as overconfident."""

_DECIMALS = 6


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, _DECIMALS)


def _ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else _round(numerator / denominator)


@dataclass(frozen=True)
class Observation:
    """One case's result for one predicate, with its label."""

    case_id: str
    result: Result
    expected: Expectation
    labeled: bool
    """Whether the case labels this predicate at all (``expected`` may still be ``None`` = ambiguous)."""


# ---- nouls ----------------------------------------------------------------------------------------


class Confusion(BaseModel):
    """Counts and rates at one threshold."""

    model_config = ConfigDict(frozen=True)

    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int
    accuracy: float | None
    precision: float | None
    recall: float | None
    f1: float | None
    fpr: float | None


class CalibrationBin(BaseModel):
    model_config = ConfigDict(frozen=True)

    lower: float
    upper: float
    n: int
    mean_probability: float | None
    positive_rate: float | None


class Recommendation(BaseModel):
    """A threshold chosen by a stated objective over the sweep."""

    model_config = ConfigDict(frozen=True)

    objective: str
    threshold: float | None
    value: float | None
    precision: float | None = None
    recall: float | None = None
    fpr: float | None = None


def confusion(probabilities: Sequence[float], labels: Sequence[bool], threshold: float) -> Confusion:
    """Counts and rates when ``probability >= threshold`` is predicted true."""
    tp = fp = tn = fn = 0
    for probability, label in zip(probabilities, labels, strict=True):
        predicted = probability >= threshold
        if predicted and label:
            tp += 1
        elif predicted:
            fp += 1
        elif label:
            fn += 1
        else:
            tn += 1
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = _round(2 * precision * recall / (precision + recall))
    return Confusion(
        threshold=threshold,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        accuracy=_ratio(tp + tn, tp + fp + tn + fn),
        precision=precision,
        recall=recall,
        f1=f1,
        fpr=_ratio(fp, fp + tn),
    )


def roc_auc(probabilities: Sequence[float], labels: Sequence[bool]) -> float | None:
    """Probability that a random positive outranks a random negative (ties count half)."""
    positives = [p for p, label in zip(probabilities, labels, strict=True) if label]
    negatives = [p for p, label in zip(probabilities, labels, strict=True) if not label]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return _round(wins / (len(positives) * len(negatives)))


def average_precision(probabilities: Sequence[float], labels: Sequence[bool]) -> float | None:
    """Area under the precision-recall curve as average precision over distinct thresholds."""
    total_positives = sum(labels)
    if total_positives == 0:
        return None
    ranked = sorted(zip(probabilities, labels, strict=True), key=lambda pair: -pair[0])
    area = 0.0
    tp = fp = 0
    previous_recall = 0.0
    index = 0
    while index < len(ranked):
        score = ranked[index][0]
        while index < len(ranked) and ranked[index][0] == score:
            if ranked[index][1]:
                tp += 1
            else:
                fp += 1
            index += 1
        recall = tp / total_positives
        precision = tp / (tp + fp)
        area += (recall - previous_recall) * precision
        previous_recall = recall
    return _round(area)


def brier(probabilities: Sequence[float], labels: Sequence[bool]) -> float | None:
    if not probabilities:
        return None
    return _round(
        sum((p - float(y)) ** 2 for p, y in zip(probabilities, labels, strict=True)) / len(probabilities)
    )


def calibration(
    probabilities: Sequence[float], labels: Sequence[bool]
) -> tuple[list[CalibrationBin], float | None]:
    """Equal-width reliability bins and expected calibration error (``None`` under 10 cases)."""
    bins: list[CalibrationBin] = []
    ece = 0.0
    n = len(probabilities)
    for index in range(ECE_BINS):
        lower = index / ECE_BINS
        upper = (index + 1) / ECE_BINS
        members = [
            (p, y)
            for p, y in zip(probabilities, labels, strict=True)
            if lower <= p < upper or (index == ECE_BINS - 1 and p == 1.0)
        ]
        if members:
            mean_p = sum(p for p, _ in members) / len(members)
            rate = sum(1 for _, y in members if y) / len(members)
            ece += len(members) / n * abs(mean_p - rate)
            bins.append(
                CalibrationBin(
                    lower=lower,
                    upper=upper,
                    n=len(members),
                    mean_probability=_round(mean_p),
                    positive_rate=_round(rate),
                )
            )
        else:
            bins.append(
                CalibrationBin(lower=lower, upper=upper, n=0, mean_probability=None, positive_rate=None)
            )
    return bins, (_round(ece) if n >= MIN_ECE_CASES else None)


def best_f1(sweep: Sequence[Confusion]) -> Recommendation:
    candidates = [row for row in sweep if row.f1 is not None]
    if not candidates:
        return Recommendation(objective="max F1", threshold=None, value=None)
    best = max(candidates, key=lambda row: (row.f1 or 0.0, row.precision or 0.0, -row.threshold))
    return Recommendation(
        objective="max F1",
        threshold=best.threshold,
        value=best.f1,
        precision=best.precision,
        recall=best.recall,
        fpr=best.fpr,
    )


def best_recall_at_fpr(sweep: Sequence[Confusion], max_fpr: float) -> Recommendation:
    objective = f"max recall with false-positive rate <= {max_fpr:g}"
    candidates = [
        row for row in sweep if row.fpr is not None and row.fpr <= max_fpr and row.recall is not None
    ]
    if not candidates:
        return Recommendation(objective=objective, threshold=None, value=None)
    best = max(candidates, key=lambda row: (row.recall or 0.0, -(row.fpr or 0.0), row.threshold))
    return Recommendation(
        objective=objective,
        threshold=best.threshold,
        value=best.recall,
        precision=best.precision,
        recall=best.recall,
        fpr=best.fpr,
    )


class NoulMetrics(BaseModel):
    """Quality of a yes/no predicate."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["noul"] = "noul"
    evaluated: int
    labeled: int
    positives: int
    negatives: int
    ambiguous: int
    threshold: float
    threshold_source: Literal["predicate", "default", "override"]
    at_threshold: Confusion
    sweep: list[Confusion]
    roc_auc: float | None
    pr_auc: float | None
    brier: float | None
    ece: float | None
    calibration: list[CalibrationBin]
    best_f1: Recommendation
    best_recall_at_fpr: Recommendation
    fragile: list[str]
    """Labeled cases within ``FRAGILITY_MARGIN`` of the operating threshold, where repeat noise matters."""
    ambiguous_mean: float | None
    ambiguous_min: float | None
    ambiguous_max: float | None
    ambiguous_in_band: int
    """Ambiguous cases with a probability inside the 0.3 to 0.7 band."""
    ambiguous_confident: list[str]
    """Ambiguous cases the model answered with near-certainty."""
    notes: list[str]

    def headline(self) -> tuple[str, float | None]:
        return "f1", self.at_threshold.f1


def noul_metrics(
    observations: Sequence[Observation],
    *,
    threshold: float,
    threshold_source: Literal["predicate", "default", "override"],
    max_fpr: float,
) -> NoulMetrics:
    scored = [
        o
        for o in observations
        if o.labeled and isinstance(o.expected, bool) and isinstance(o.result, NoulResult)
    ]
    probabilities = [o.result.probability for o in scored if isinstance(o.result, NoulResult)]
    labels = [bool(o.expected) for o in scored]
    ambiguous = [
        o.result.probability
        for o in observations
        if o.labeled and o.expected is None and isinstance(o.result, NoulResult)
    ]
    ambiguous_ids = [
        o.case_id
        for o in observations
        if o.labeled and o.expected is None and isinstance(o.result, NoulResult)
    ]
    sweep = [confusion(probabilities, labels, t) for t in SWEEP_THRESHOLDS]
    bins, ece = calibration(probabilities, labels)
    notes: list[str] = []
    positives, negatives = sum(labels), len(labels) - sum(labels)
    if positives == 0 or negatives == 0:
        notes.append("ROC-AUC is undefined: only one class is labeled")
    if positives == 0:
        notes.append("recall and PR-AUC are undefined: no case is labeled true")
    if len(probabilities) < MIN_ECE_CASES:
        notes.append(f"ECE is undefined under {MIN_ECE_CASES} labeled cases")
    elif len(probabilities) < 100:
        notes.append("calibration estimates are rough under 100 labeled cases")
    fragile = [
        o.case_id
        for o in scored
        if isinstance(o.result, NoulResult) and abs(o.result.probability - threshold) < FRAGILITY_MARGIN
    ]
    return NoulMetrics(
        evaluated=len([o for o in observations if isinstance(o.result, NoulResult)]),
        labeled=len(scored),
        positives=positives,
        negatives=negatives,
        ambiguous=len(ambiguous),
        threshold=threshold,
        threshold_source=threshold_source,
        at_threshold=confusion(probabilities, labels, threshold),
        sweep=sweep,
        roc_auc=roc_auc(probabilities, labels),
        pr_auc=average_precision(probabilities, labels),
        brier=brier(probabilities, labels),
        ece=ece,
        calibration=bins,
        best_f1=best_f1(sweep),
        best_recall_at_fpr=best_recall_at_fpr(sweep, max_fpr),
        fragile=fragile,
        ambiguous_mean=_round(sum(ambiguous) / len(ambiguous)) if ambiguous else None,
        ambiguous_min=min(ambiguous) if ambiguous else None,
        ambiguous_max=max(ambiguous) if ambiguous else None,
        ambiguous_in_band=sum(1 for p in ambiguous if AMBIGUOUS_BAND[0] <= p <= AMBIGUOUS_BAND[1]),
        ambiguous_confident=[
            case_id
            for case_id, p in zip(ambiguous_ids, ambiguous, strict=True)
            if p < CONFIDENT or p > 1 - CONFIDENT
        ],
        notes=notes,
    )


# ---- choices --------------------------------------------------------------------------------------


class OptionMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    support: int
    """Single-label cases expecting this option."""
    predicted: int
    precision: float | None
    recall: float | None


class BucketMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    lower: float
    upper: float
    n: int
    accuracy: float | None


class AbstentionRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    min_confidence: float
    coverage: float | None
    """Share of labeled cases answered at or above the confidence threshold."""
    accuracy: float | None
    """Accuracy among the answered cases."""


def _choice_correct(choice_value: str, expected: Expectation) -> bool:
    return choice_value in expected if isinstance(expected, list) else choice_value == expected


def _buckets(pairs: Sequence[tuple[float, bool]]) -> list[BucketMetrics]:
    rows: list[BucketMetrics] = []
    for index, (lower, upper) in enumerate(CONFIDENCE_BUCKETS):
        last = index == len(CONFIDENCE_BUCKETS) - 1
        members = [
            correct
            for confidence, correct in pairs
            if lower <= confidence < upper or (last and confidence == upper)
        ]
        rows.append(
            BucketMetrics(
                lower=lower, upper=upper, n=len(members), accuracy=_ratio(sum(members), len(members))
            )
        )
    return rows


class ChoiceMetrics(BaseModel):
    """Quality of a choice predicate."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["choice"] = "choice"
    evaluated: int
    labeled: int
    ambiguous: int
    accuracy: float | None
    options: dict[str, OptionMetrics]
    confusion: dict[str, dict[str, int]]
    """Expected option to predicted option counts, single-label cases only."""
    by_confidence: list[BucketMetrics]
    abstention: list[AbstentionRow]
    min_confidence: float | None
    at_min_confidence: AbstentionRow | None
    notes: list[str]

    def headline(self) -> tuple[str, float | None]:
        return "accuracy", self.accuracy


def choice_metrics(
    observations: Sequence[Observation], *, options: Sequence[str], min_confidence: float | None
) -> ChoiceMetrics:
    scored = [
        o for o in observations if o.labeled and o.expected is not None and isinstance(o.result, ChoiceResult)
    ]
    outcomes = [
        (o.result, _choice_correct(o.result.choice, o.expected))
        for o in scored
        if isinstance(o.result, ChoiceResult)
    ]
    single = [
        (o.result, o.expected)
        for o in scored
        if isinstance(o.result, ChoiceResult) and isinstance(o.expected, str)
    ]
    per_option: dict[str, OptionMetrics] = {}
    matrix: dict[str, dict[str, int]] = {option: dict.fromkeys(options, 0) for option in options}
    for result, expected in single:
        assert isinstance(expected, str)
        matrix[expected][result.choice] += 1
    for option in options:
        support = sum(1 for _, expected in single if expected == option)
        predicted = sum(1 for result, _ in single if result.choice == option)
        hits = matrix[option][option]
        per_option[option] = OptionMetrics(
            support=support,
            predicted=predicted,
            precision=_ratio(hits, predicted),
            recall=_ratio(hits, support),
        )

    def abstain(threshold: float) -> AbstentionRow:
        answered = [correct for result, correct in outcomes if result.confidence >= threshold]
        return AbstentionRow(
            min_confidence=threshold,
            coverage=_ratio(len(answered), len(outcomes)),
            accuracy=_ratio(sum(answered), len(answered)),
        )

    notes: list[str] = []
    if len(single) < len(scored):
        notes.append(
            f"{len(scored) - len(single)} case(s) accept several options and are excluded from the "
            "confusion matrix"
        )
    return ChoiceMetrics(
        evaluated=len([o for o in observations if isinstance(o.result, ChoiceResult)]),
        labeled=len(scored),
        ambiguous=len([o for o in observations if o.labeled and o.expected is None]),
        accuracy=_ratio(sum(correct for _, correct in outcomes), len(outcomes)),
        options=per_option,
        confusion=matrix,
        by_confidence=_buckets([(result.confidence, correct) for result, correct in outcomes]),
        abstention=[abstain(threshold) for threshold in ABSTENTION_THRESHOLDS],
        min_confidence=min_confidence,
        at_min_confidence=abstain(min_confidence) if min_confidence is not None else None,
        notes=notes,
    )


# ---- scores ---------------------------------------------------------------------------------------


class ScoreMetrics(BaseModel):
    """Quality of a score predicate."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["score"] = "score"
    evaluated: int
    exact_labeled: int
    range_labeled: int
    ambiguous: int
    exact_accuracy: float | None
    """Share of exact-level cases whose nearest level matches."""
    within_one: float | None
    mae: float | None
    """Mean absolute error in levels between the score and the expected level."""
    range_satisfaction: float | None
    """Share of range cases whose nearest level lies in the range."""
    range_mae: float | None
    """Mean distance in levels from the score to the scores that round into the range (0 inside)."""
    by_confidence: list[BucketMetrics]
    notes: list[str]

    def headline(self) -> tuple[str, float | None]:
        if self.exact_labeled:
            return "exact_accuracy", self.exact_accuracy
        return "range_satisfaction", self.range_satisfaction


def score_correct(result: ScoreResult, expected: Expectation) -> bool:
    if isinstance(expected, ScoreRange):
        return expected.contains(result.nearest_level)
    return result.nearest_level == expected


def score_metrics(observations: Sequence[Observation]) -> ScoreMetrics:
    exact: list[tuple[ScoreResult, int]] = []
    ranged: list[tuple[ScoreResult, ScoreRange]] = []
    for observation in observations:
        result, expected = observation.result, observation.expected
        if not observation.labeled or not isinstance(result, ScoreResult):
            continue
        if isinstance(expected, ScoreRange):
            ranged.append((result, expected))
        elif isinstance(expected, int) and not isinstance(expected, bool):
            exact.append((result, expected))
    errors = [abs(result.score - expected) for result, expected in exact]
    distances = [expected.distance(result.score) for result, expected in ranged]
    pairs = [(result.confidence, result.nearest_level == expected) for result, expected in exact]
    pairs += [(result.confidence, expected.contains(result.nearest_level)) for result, expected in ranged]
    return ScoreMetrics(
        evaluated=len([o for o in observations if isinstance(o.result, ScoreResult)]),
        exact_labeled=len(exact),
        range_labeled=len(ranged),
        ambiguous=len([o for o in observations if o.labeled and o.expected is None]),
        exact_accuracy=_ratio(
            sum(1 for result, expected in exact if result.nearest_level == expected), len(exact)
        ),
        within_one=_ratio(
            sum(1 for result, expected in exact if abs(result.nearest_level - expected) <= 1), len(exact)
        ),
        mae=_round(sum(errors) / len(errors)) if errors else None,
        range_satisfaction=_ratio(
            sum(1 for result, expected in ranged if expected.contains(result.nearest_level)), len(ranged)
        ),
        range_mae=_round(sum(distances) / len(distances)) if distances else None,
        by_confidence=_buckets(pairs),
        notes=[],
    )


PredicateMetrics = Annotated[NoulMetrics | ChoiceMetrics | ScoreMetrics, Field(discriminator="kind")]


def percentile(values: Sequence[float], fraction: float) -> float | None:
    """Nearest-rank percentile (``fraction`` in 0..1)."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[rank - 1]
