"""Compare an evaluation report against a baseline.

The regression gate fails when:

- the predicate set's fingerprint differs from the baseline's (refresh, review, then accept);
- the resolved model changed, unless the comparison is explicitly a model upgrade;
- any predicate's headline metric (noul F1 at its threshold, choice accuracy, score accuracy) drops
  by more than the tolerance, or becomes undefined;
- a case that was correct in the baseline is now wrong, and the answer is not fragile.

An answer is fragile when it sits within the tolerance of its decision boundary in either report:
a noul probability near its threshold, a choice whose top two options are within twice the
tolerance, or a score near a rounding boundary. Repeat noise of that size is expected (see
``sentinel eval stability``), so fragile flips are listed but do not fail the gate.
"""

from pydantic import BaseModel, ConfigDict

from sentinel_runtime.evals.fixtures import Expectation
from sentinel_runtime.evals.report import EvalReport, Prediction

DEFAULT_TOLERANCE = 0.05


class MetricChange(BaseModel):
    model_config = ConfigDict(frozen=True)

    predicate: str
    metric: str
    baseline: float | None
    current: float | None
    delta: float | None


class CaseChange(BaseModel):
    """A labeled prediction whose correctness changed between the reports."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    predicate: str
    expected: Expectation
    baseline: float | str
    current: float | str
    now_correct: bool
    fragile: bool


class ProbabilityShift(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    predicate: str
    baseline: float
    current: float
    delta: float


class Comparison(BaseModel):
    """The difference between two reports and the regression verdict."""

    model_config = ConfigDict(frozen=True, title="Sentinel report comparison (v1)")

    tolerance: float
    baseline_fingerprint: str
    current_fingerprint: str
    baseline_models: list[str]
    current_models: list[str]
    model_upgrade: bool
    metrics: list[MetricChange]
    changes: list[CaseChange]
    shifts: list[ProbabilityShift]
    """Noul probabilities that moved by more than the tolerance."""
    cases_added: list[str]
    cases_removed: list[str]
    regressions: list[str]

    @property
    def passed(self) -> bool:
        return not self.regressions


def _fragile(prediction: Prediction, threshold: float | None, tolerance: float) -> bool:
    """Whether ``prediction`` sits within repeat noise of its decision boundary."""
    if prediction.kind == "noul" and threshold is not None and isinstance(prediction.value, float | int):
        return abs(float(prediction.value) - threshold) <= tolerance
    if prediction.kind == "score" and isinstance(prediction.value, float | int):
        return abs(float(prediction.value) % 1.0 - 0.5) <= tolerance
    if (
        prediction.kind == "choice"
        and prediction.probabilities is not None
        and len(prediction.probabilities) >= 2
    ):
        top, second = sorted(prediction.probabilities.values(), reverse=True)[:2]
        return top - second <= 2 * tolerance
    return False


def compare_reports(
    baseline: EvalReport,
    current: EvalReport,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    allow_model_change: bool = False,
) -> Comparison:
    """Compare ``current`` against ``baseline`` and list regressions."""
    regressions: list[str] = []
    if baseline.predicate_set_fingerprint != current.predicate_set_fingerprint:
        regressions.append(
            "the predicate set changed since the baseline; run the suite with --refresh, review the report, "
            "then `sentinel eval accept` it"
        )
    model_upgrade = baseline.models_resolved != current.models_resolved
    if model_upgrade and not allow_model_change:
        regressions.append(
            f"the model changed from {', '.join(baseline.models_resolved) or 'none'} to "
            f"{', '.join(current.models_resolved) or 'none'}; pass --allow-model-change to compare "
            "across models"
        )

    metrics: list[MetricChange] = []
    for predicate_id, current_metrics in current.predicates.items():
        name, value = current_metrics.headline()
        previous = baseline.predicates.get(predicate_id)
        before = (
            previous.headline()[1] if previous is not None and previous.kind == current_metrics.kind else None
        )
        delta = None if before is None or value is None else round(value - before, 6)
        metrics.append(
            MetricChange(predicate=predicate_id, metric=name, baseline=before, current=value, delta=delta)
        )
        if previous is None:
            continue
        if before is not None and value is None:
            regressions.append(f"{predicate_id} {name} became undefined (was {before:.2f})")
        elif delta is not None and delta < -tolerance:
            regressions.append(f"{predicate_id} {name} dropped from {before:.2f} to {value:.2f}")

    baseline_cases = {record.id: record for record in baseline.cases}
    current_cases = {record.id: record for record in current.cases}
    changes: list[CaseChange] = []
    shifts: list[ProbabilityShift] = []
    for case_id, record in current_cases.items():
        previous_record = baseline_cases.get(case_id)
        if previous_record is None:
            continue
        for predicate_id, prediction in record.predictions.items():
            previous = previous_record.predictions.get(predicate_id)
            if previous is None:
                continue
            if (
                prediction.kind == "noul"
                and isinstance(prediction.value, float | int)
                and isinstance(previous.value, float | int)
            ):
                moved = round(float(prediction.value) - float(previous.value), 6)
                if abs(moved) > tolerance:
                    shifts.append(
                        ProbabilityShift(
                            case_id=case_id,
                            predicate=predicate_id,
                            baseline=float(previous.value),
                            current=float(prediction.value),
                            delta=moved,
                        )
                    )
            if (
                previous.correct is None
                or prediction.correct is None
                or previous.correct == prediction.correct
            ):
                continue
            fragile = _fragile(prediction, current.thresholds.get(predicate_id), tolerance) or _fragile(
                previous, baseline.thresholds.get(predicate_id), tolerance
            )
            change = CaseChange(
                case_id=case_id,
                predicate=predicate_id,
                expected=prediction.expected,
                baseline=previous.value,
                current=prediction.value,
                now_correct=prediction.correct,
                fragile=fragile,
            )
            changes.append(change)
            if not change.now_correct and not change.fragile:
                regressions.append(f"{case_id}: {predicate_id} was correct and is now wrong")

    return Comparison(
        tolerance=tolerance,
        baseline_fingerprint=baseline.predicate_set_fingerprint,
        current_fingerprint=current.predicate_set_fingerprint,
        baseline_models=baseline.models_resolved,
        current_models=current.models_resolved,
        model_upgrade=model_upgrade,
        metrics=metrics,
        changes=changes,
        shifts=shifts,
        cases_added=sorted(set(current_cases) - set(baseline_cases)),
        cases_removed=sorted(set(baseline_cases) - set(current_cases)),
        regressions=regressions,
    )
