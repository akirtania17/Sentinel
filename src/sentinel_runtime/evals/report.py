"""Evaluation reports: build from runs, serialize to JSON, render for people.

A report records what was measured (suite, predicate set and fingerprints, requested and resolved
models, filters, operating thresholds), the metrics per predicate and per tag, usage, and every case's
prediction next to its label, so any number in the report can be traced back to cases.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from sentinel_runtime.errors import ErrorInfo, FixtureValidationError
from sentinel_runtime.evals.fixtures import Case, EvalSuite, Expectation, ScoreRange
from sentinel_runtime.evals.metrics import (
    ChoiceMetrics,
    NoulMetrics,
    Observation,
    PredicateMetrics,
    ScoreMetrics,
    choice_metrics,
    noul_metrics,
    percentile,
    score_correct,
    score_metrics,
)
from sentinel_runtime.predicates import ChoicePredicate, NoulPredicate, Predicate
from sentinel_runtime.results import ChoiceResult, Evaluation, NoulResult, Result, ScoreResult

REPORT_FORMAT = "sentinel/eval-report/v1"
DEFAULT_THRESHOLD = 0.5
MIN_SLICE_CASES = 5
MAX_MISSES_SHOWN = 15


@dataclass(frozen=True)
class CaseRun:
    """The outcome of evaluating one case."""

    case: Case
    evaluation: Evaluation | None
    error: ErrorInfo | None = None


class Prediction(BaseModel):
    """One predicate's answer on one case, next to its label."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["noul", "choice", "score"]
    value: float | str
    """Probability (noul), chosen option (choice), or score (score)."""
    confidence: float | None = None
    probabilities: dict[str, float] | None = None
    """The full distribution for a choice."""
    nearest_level: int | None = None
    labeled: bool
    expected: Expectation = None
    correct: bool | None = None
    """``None`` when the case leaves this predicate unlabeled or marks it ambiguous."""


class CaseRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    tags: list[str]
    source: str
    note: str | None = None
    error: ErrorInfo | None = None
    predictions: dict[str, Prediction]


class SliceMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    labeled: int
    metric: str
    value: float | None


class UsageSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    evaluations: int
    requests: int
    input_tokens: int | None
    estimated_cost_usd: float | None
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    replayed: int
    """Evaluations served from the response cache; their latency is the recorded original latency."""


class EvalReport(BaseModel):
    """The result of running a fixture suite."""

    model_config = ConfigDict(frozen=True, title="Sentinel evaluation report (v1)")

    format: Literal["sentinel/eval-report/v1"] = "sentinel/eval-report/v1"
    sentinel_version: str
    created_at: datetime
    suite: str
    predicate_set: str
    predicate_set_fingerprint: str
    predicate_fingerprints: dict[str, str]
    model_requested: str
    models_resolved: list[str]
    cases_total: int
    cases_selected: int
    cases_evaluated: int
    cases_failed: int
    unreviewed_excluded: int
    tags: list[str] | None = None
    exclude_tags: list[str] | None = None
    thresholds: dict[str, float]
    """Operating thresholds for noul predicates."""
    max_fpr: float
    predicates: dict[str, PredicateMetrics]
    slices: dict[str, dict[str, SliceMetrics]]
    usage: UsageSummary
    cases: list[CaseRecord]

    def headline(self, predicate_id: str) -> tuple[str, float | None]:
        return self.predicates[predicate_id].headline()

    @classmethod
    def load(cls, path: str | Path) -> "EvalReport":
        try:
            return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
        except OSError as error:
            raise FixtureValidationError(f"cannot read report: {error.strerror}", path=str(path)) from None
        except ValidationError as error:
            raise FixtureValidationError(
                f"not a Sentinel evaluation report ({error.errors()[0]['msg']})", path=str(path)
            ) from None

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")


# ---- building -------------------------------------------------------------------------------------


def operating_thresholds(
    predicates: Sequence[Predicate], overrides: Mapping[str, float]
) -> dict[str, tuple[float, str]]:
    thresholds: dict[str, tuple[float, str]] = {}
    for predicate in predicates:
        if not isinstance(predicate, NoulPredicate):
            continue
        if predicate.id in overrides:
            thresholds[predicate.id] = (overrides[predicate.id], "override")
        elif predicate.threshold is not None:
            thresholds[predicate.id] = (predicate.threshold, "predicate")
        else:
            thresholds[predicate.id] = (DEFAULT_THRESHOLD, "default")
    return thresholds


def _prediction(result: Result, case: Case, predicate_id: str, threshold: float | None) -> Prediction:
    labeled = predicate_id in case.expect
    expected = case.expect.get(predicate_id)
    scored = labeled and expected is not None
    if isinstance(result, NoulResult):
        assert threshold is not None
        correct = ((result.probability >= threshold) == expected) if scored else None
        return Prediction(
            kind="noul", value=result.probability, labeled=labeled, expected=expected, correct=correct
        )
    if isinstance(result, ChoiceResult):
        correct = (
            (result.choice in expected if isinstance(expected, list) else result.choice == expected)
            if scored
            else None
        )
        return Prediction(
            kind="choice",
            value=result.choice,
            confidence=result.confidence,
            probabilities=dict(result.probabilities),
            labeled=labeled,
            expected=expected,
            correct=correct,
        )
    assert isinstance(result, ScoreResult)
    return Prediction(
        kind="score",
        value=result.score,
        confidence=result.confidence,
        nearest_level=result.nearest_level,
        labeled=labeled,
        expected=expected,
        correct=score_correct(result, expected) if scored else None,
    )


def _metrics_for(
    predicate: Predicate,
    observations: Sequence[Observation],
    thresholds: Mapping[str, tuple[float, str]],
    max_fpr: float,
) -> PredicateMetrics:
    if isinstance(predicate, NoulPredicate):
        threshold, source = thresholds[predicate.id]
        return noul_metrics(observations, threshold=threshold, threshold_source=source, max_fpr=max_fpr)  # type: ignore[arg-type]
    if isinstance(predicate, ChoicePredicate):
        return choice_metrics(
            observations, options=list(predicate.options), min_confidence=predicate.min_confidence
        )
    return score_metrics(observations)


def build_report(
    suite: EvalSuite,
    runs: Sequence[CaseRun],
    *,
    model_requested: str,
    sentinel_version: str,
    threshold_overrides: Mapping[str, float] | None = None,
    max_fpr: float = 0.05,
    tags: Sequence[str] | None = None,
    exclude_tags: Sequence[str] | None = None,
    unreviewed_excluded: int = 0,
) -> EvalReport:
    """Score ``runs`` against their labels and assemble the report."""
    predicates = suite.predicate_set.predicates
    thresholds = operating_thresholds(predicates, threshold_overrides or {})
    unknown = set(threshold_overrides or {}) - set(thresholds)
    if unknown:
        raise FixtureValidationError(
            f"--threshold names non-noul or unknown predicate(s): {', '.join(sorted(unknown))}"
        )

    observations: dict[str, list[tuple[Case, Observation]]] = {p.id: [] for p in predicates}
    records: list[CaseRecord] = []
    for run in runs:
        predictions: dict[str, Prediction] = {}
        results = run.evaluation.results if run.evaluation is not None else {}
        for predicate in predicates:
            result = results.get(predicate.id)
            if result is None:
                continue
            threshold = thresholds.get(predicate.id, (None, ""))[0]
            predictions[predicate.id] = _prediction(result, run.case, predicate.id, threshold)
            observation = Observation(
                case_id=run.case.id,
                result=result,
                expected=run.case.expect.get(predicate.id),
                labeled=predicate.id in run.case.expect,
            )
            observations[predicate.id].append((run.case, observation))
        records.append(
            CaseRecord(
                id=run.case.id,
                tags=list(run.case.tags),
                source=run.case.source,
                note=run.case.note,
                error=run.error,
                predictions=predictions,
            )
        )

    metrics = {
        p.id: _metrics_for(p, [o for _, o in observations[p.id]], thresholds, max_fpr) for p in predicates
    }
    all_tags = sorted({tag for run in runs for tag in run.case.tags})
    slices: dict[str, dict[str, SliceMetrics]] = {}
    for tag in all_tags:
        slices[tag] = {}
        for predicate in predicates:
            subset = [o for case, o in observations[predicate.id] if tag in case.tags]
            sliced = _metrics_for(predicate, subset, thresholds, max_fpr)
            name, value = sliced.headline()
            labeled = (
                sliced.labeled
                if isinstance(sliced, NoulMetrics | ChoiceMetrics)
                else sliced.exact_labeled + sliced.range_labeled
            )
            slices[tag][predicate.id] = SliceMetrics(labeled=labeled, metric=name, value=value)

    evaluations = [run.evaluation for run in runs if run.evaluation is not None]
    calls = [call for evaluation in evaluations for call in evaluation.calls]
    tokens = [call.input_tokens for call in calls]
    costs = [evaluation.usage.estimated_cost_usd for evaluation in evaluations]
    latencies = [
        evaluation.latency_ms
        if not evaluation.usage.replayed
        else sum(c.latency_ms for c in evaluation.calls)
        for evaluation in evaluations
    ]
    usage = UsageSummary(
        evaluations=len(evaluations),
        requests=len(calls),
        input_tokens=None if any(t is None for t in tokens) else sum(t for t in tokens if t is not None),
        estimated_cost_usd=None
        if any(c is None for c in costs)
        else round(sum(c for c in costs if c is not None), 8),
        latency_p50_ms=percentile(latencies, 0.5),
        latency_p95_ms=percentile(latencies, 0.95),
        replayed=sum(1 for evaluation in evaluations if evaluation.usage.replayed),
    )
    return EvalReport(
        sentinel_version=sentinel_version,
        created_at=datetime.now(UTC).replace(microsecond=0),
        suite=suite.name,
        predicate_set=suite.predicate_set.name,
        predicate_set_fingerprint=suite.predicate_set.fingerprint,
        predicate_fingerprints=suite.predicate_set.predicate_fingerprints(),
        model_requested=model_requested,
        models_resolved=sorted(
            {model for e in evaluations for model in e.model_resolved.split(",") if model}
        ),
        cases_total=len(suite.cases),
        cases_selected=len(runs),
        cases_evaluated=len([run for run in runs if run.error is None and run.evaluation is not None]),
        cases_failed=len([run for run in runs if run.error is not None]),
        unreviewed_excluded=unreviewed_excluded,
        tags=list(tags) if tags else None,
        exclude_tags=list(exclude_tags) if exclude_tags else None,
        thresholds={pid: value for pid, (value, _) in thresholds.items()},
        max_fpr=max_fpr,
        predicates=metrics,
        slices=slices,
        usage=usage,
        cases=records,
    )


# ---- rendering ------------------------------------------------------------------------------------


def fmt(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _expected_text(expected: Expectation) -> str:
    if isinstance(expected, ScoreRange):
        low = "" if expected.min is None else str(expected.min)
        high = "" if expected.max is None else str(expected.max)
        return f"level {low}..{high}"
    if isinstance(expected, list):
        return " or ".join(expected)
    return str(expected).lower() if isinstance(expected, bool) else str(expected)


def _got_text(prediction: Prediction) -> str:
    if prediction.kind == "noul":
        return f"p {fmt(float(prediction.value))}"
    if prediction.kind == "choice":
        return f"{prediction.value} (confidence {fmt(prediction.confidence)})"
    return f"{fmt(float(prediction.value))} (level {prediction.nearest_level})"


def _detail(metrics: PredicateMetrics) -> str:
    if isinstance(metrics, NoulMetrics):
        at = metrics.at_threshold
        return (
            f"precision {fmt(at.precision)}, recall {fmt(at.recall)}, ROC-AUC {fmt(metrics.roc_auc)}, "
            f"PR-AUC {fmt(metrics.pr_auc)}, Brier {fmt(metrics.brier, 3)}, ECE {fmt(metrics.ece, 3)}, "
            f"fragile {len(metrics.fragile)}"
        )
    if isinstance(metrics, ChoiceMetrics):
        coverage = ""
        if metrics.at_min_confidence is not None:
            row = metrics.at_min_confidence
            coverage = (
                f", at min confidence {fmt(metrics.min_confidence)}: coverage {fmt(row.coverage)}, "
                f"accuracy {fmt(row.accuracy)}"
            )
        return f"{len(metrics.options)} options{coverage}"
    return (
        f"within one level {fmt(metrics.within_one)}, MAE {fmt(metrics.mae)}, "
        f"ranges satisfied {fmt(metrics.range_satisfaction)} of {metrics.range_labeled}"
    )


def _metric_label(metrics: PredicateMetrics) -> str:
    if isinstance(metrics, NoulMetrics):
        return f"F1 at {metrics.threshold:.2f}"
    name, _ = metrics.headline()
    return name.replace("_", " ")


def _labeled(metrics: PredicateMetrics) -> int:
    if isinstance(metrics, NoulMetrics | ChoiceMetrics):
        return metrics.labeled
    return metrics.exact_labeled + metrics.range_labeled


def misses(report: EvalReport) -> list[tuple[str, str, Prediction]]:
    """Every labeled prediction that was wrong, as ``(case id, predicate id, prediction)``."""
    return [
        (record.id, predicate_id, prediction)
        for record in report.cases
        for predicate_id, prediction in record.predictions.items()
        if prediction.correct is False
    ]


def render_report(report: EvalReport, console: Console, *, details: bool = False) -> None:
    """Print a report summary (and, with ``details``, sweeps, matrices and calibration)."""
    models = ", ".join(report.models_resolved) or "none"
    console.print(
        f"[bold]{escape(report.suite)}[/]  {report.predicate_set} {report.predicate_set_fingerprint[:23]}  "
        f"model {escape(models)} (requested {escape(report.model_requested)})"
    )
    counted = f"{report.cases_evaluated} of {report.cases_selected} cases evaluated"
    if report.cases_failed:
        counted += f", {report.cases_failed} failed"
    if report.unreviewed_excluded:
        counted += f", {report.unreviewed_excluded} unreviewed excluded"
    console.print(counted, markup=False)

    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    for column in ("predicate", "kind", "labeled", "metric", "value", "detail"):
        table.add_column(column, no_wrap=column != "detail")
    for predicate_id, metrics in report.predicates.items():
        _, value = metrics.headline()
        table.add_row(
            predicate_id,
            metrics.kind,
            str(_labeled(metrics)),
            _metric_label(metrics),
            fmt(value),
            _detail(metrics),
        )
    console.print(table)

    nouls = {pid: m for pid, m in report.predicates.items() if isinstance(m, NoulMetrics)}
    if nouls:
        console.print("[bold]Thresholds are empirical: pick them per application risk.[/]")
        for predicate_id, metrics in nouls.items():
            best, safe = metrics.best_f1, metrics.best_recall_at_fpr
            console.print(
                f"  {predicate_id}: {best.objective} at {fmt(best.threshold)} (F1 {fmt(best.value)}); "
                f"{safe.objective} at {fmt(safe.threshold)} (recall {fmt(safe.value)})",
                markup=False,
            )

    if report.slices:
        slice_table = Table(
            box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False, title="by tag", title_justify="left"
        )
        slice_table.add_column("tag", no_wrap=True)
        for predicate_id in report.predicates:
            slice_table.add_column(predicate_id, no_wrap=True)
        for tag, row in report.slices.items():
            cells = [tag]
            for predicate_id in report.predicates:
                entry = row[predicate_id]
                if entry.labeled == 0:
                    cells.append("-")
                elif entry.labeled < MIN_SLICE_CASES:
                    cells.append(f"n={entry.labeled}")
                else:
                    cells.append(f"{fmt(entry.value)} (n={entry.labeled})")
            slice_table.add_row(*cells)
        console.print(slice_table)

    wrong = misses(report)
    if wrong:
        console.print(f"[bold]misses[/] ({len(wrong)})")
        for case_id, predicate_id, prediction in wrong[:MAX_MISSES_SHOWN]:
            expected, got = _expected_text(prediction.expected), _got_text(prediction)
            console.print(f"  {case_id}  {predicate_id}: expected {expected}, got {got}", markup=False)
        if len(wrong) > MAX_MISSES_SHOWN:
            console.print(
                f"  ... and {len(wrong) - MAX_MISSES_SHOWN} more (see the JSON report)", markup=False
            )
    failures = [record for record in report.cases if record.error is not None]
    if failures:
        console.print(f"[bold red]failed cases[/] ({len(failures)})")
        for record in failures[:MAX_MISSES_SHOWN]:
            assert record.error is not None
            console.print(f"  {record.id}: {record.error.kind} {record.error.detail}", markup=False)

    usage = report.usage
    parts = [f"{usage.requests} requests"]
    if usage.input_tokens is not None:
        parts.append(f"{usage.input_tokens:,} input tokens")
    if usage.estimated_cost_usd is not None:
        parts.append(f"${usage.estimated_cost_usd:.4f}")
    parts.append(f"latency p50 {fmt(usage.latency_p50_ms, 0)} ms, p95 {fmt(usage.latency_p95_ms, 0)} ms")
    if usage.replayed:
        parts.append(f"{usage.replayed} of {usage.evaluations} replayed from cache")
    console.print(", ".join(parts), markup=False)

    if details:
        _render_details(report, console)


def _render_details(report: EvalReport, console: Console) -> None:
    for predicate_id, metrics in report.predicates.items():
        if isinstance(metrics, NoulMetrics):
            table = Table(
                box=box.SIMPLE_HEAD, show_edge=False, title=f"{predicate_id} sweep", title_justify="left"
            )
            for column in ("threshold", "tp", "fp", "tn", "fn", "precision", "recall", "F1", "FPR"):
                table.add_column(column, justify="right")
            for row in metrics.sweep:
                table.add_row(
                    f"{row.threshold:.2f}", str(row.tp), str(row.fp), str(row.tn), str(row.fn),
                    fmt(row.precision), fmt(row.recall), fmt(row.f1), fmt(row.fpr),
                )  # fmt: skip
            console.print(table)
        elif isinstance(metrics, ChoiceMetrics):
            table = Table(
                box=box.SIMPLE_HEAD,
                show_edge=False,
                title=f"{predicate_id} confusion (expected x predicted)",
                title_justify="left",
            )
            table.add_column("expected", no_wrap=True)
            for option in metrics.confusion:
                table.add_column(option, justify="right")
            for expected, row in metrics.confusion.items():
                table.add_row(expected, *[str(count) for count in row.values()])
            console.print(table)


def _row(*cells: object) -> str:
    return "| " + " | ".join(str(cell) for cell in cells) + " |"


def _header(*columns: str) -> list[str]:
    return [_row(*columns), _row(*["---"] * len(columns))]


def _slice_cell(entry: SliceMetrics) -> str:
    if entry.labeled == 0:
        return "-"
    if entry.labeled < MIN_SLICE_CASES:
        return f"n={entry.labeled}"
    return f"{fmt(entry.value)} (n={entry.labeled})"


def _noul_markdown(metrics: NoulMetrics) -> list[str]:
    best, safe = metrics.best_f1, metrics.best_recall_at_fpr
    lines = [
        f"Operating threshold {metrics.threshold:.2f} ({metrics.threshold_source}). "
        f"{metrics.positives} labeled true, {metrics.negatives} false, {metrics.ambiguous} ambiguous.",
        "",
        f"- {best.objective}: threshold {fmt(best.threshold)}, F1 {fmt(best.value)}",
        f"- {safe.objective}: threshold {fmt(safe.threshold)}, recall {fmt(safe.value)}",
        f"- Fragile cases (within 0.10 of the threshold): {', '.join(metrics.fragile) or 'none'}",
        f"- Ambiguous cases: mean {fmt(metrics.ambiguous_mean)}, "
        f"{metrics.ambiguous_in_band} in the 0.3 to 0.7 band, answered with near-certainty: "
        f"{', '.join(metrics.ambiguous_confident) or 'none'}",
        "",
        *_header("Threshold", "TP", "FP", "TN", "FN", "Precision", "Recall", "F1", "FPR"),
    ]
    for r in metrics.sweep:
        lines.append(
            _row(
                f"{r.threshold:.2f}",
                r.tp,
                r.fp,
                r.tn,
                r.fn,
                fmt(r.precision),
                fmt(r.recall),
                fmt(r.f1),
                fmt(r.fpr),
            )
        )
    lines += ["", *_header("Probability bin", "Cases", "Mean probability", "Share true")]
    for b in metrics.calibration:
        lines.append(
            _row(f"{b.lower:.1f} to {b.upper:.1f}", b.n, fmt(b.mean_probability), fmt(b.positive_rate))
        )
    return lines


def _choice_markdown(metrics: ChoiceMetrics) -> list[str]:
    options = list(metrics.confusion)
    lines = _header("Expected \\ predicted", *options)
    lines += [_row(expected, *row.values()) for expected, row in metrics.confusion.items()]
    lines += ["", *_header("Min confidence", "Coverage", "Accuracy")]
    lines += [_row(f"{r.min_confidence:.2f}", fmt(r.coverage), fmt(r.accuracy)) for r in metrics.abstention]
    return lines


def _score_markdown(metrics: ScoreMetrics) -> list[str]:
    return [
        f"Exact labels: {metrics.exact_labeled} (accuracy {fmt(metrics.exact_accuracy)}, within one "
        f"{fmt(metrics.within_one)}, MAE {fmt(metrics.mae)}). Ranges: {metrics.range_labeled} (satisfied "
        f"{fmt(metrics.range_satisfaction)}, mean distance {fmt(metrics.range_mae)})."
    ]


def render_markdown(report: EvalReport) -> str:
    """A self-contained Markdown report with every table."""
    resolved = ", ".join(f"`{model}`" for model in report.models_resolved) or "none"
    lines = [
        f"# Evaluation: {report.suite}",
        "",
        f"- Predicate set: `{report.predicate_set}` `{report.predicate_set_fingerprint}`",
        f"- Model: requested `{report.model_requested}`, resolved {resolved}",
        f"- Cases: {report.cases_evaluated} evaluated of {report.cases_selected} selected "
        f"({report.cases_total} in suite, {report.unreviewed_excluded} unreviewed excluded, "
        f"{report.cases_failed} failed)",
        f"- Created: {report.created_at.isoformat()} with Sentinel {report.sentinel_version}",
        "",
        "## Summary",
        "",
        *_header("Predicate", "Kind", "Labeled", "Metric", "Value", "Detail"),
    ]
    for predicate_id, metrics in report.predicates.items():
        _, value = metrics.headline()
        lines.append(
            _row(
                f"`{predicate_id}`",
                metrics.kind,
                _labeled(metrics),
                _metric_label(metrics),
                fmt(value),
                _detail(metrics),
            )
        )
    for predicate_id, metrics in report.predicates.items():
        lines += ["", f"## `{predicate_id}`", ""]
        if isinstance(metrics, NoulMetrics):
            lines += _noul_markdown(metrics)
        elif isinstance(metrics, ChoiceMetrics):
            lines += _choice_markdown(metrics)
        else:
            lines += _score_markdown(metrics)
        if metrics.notes:
            lines += ["", *[f"> {note}" for note in metrics.notes]]
    if report.slices:
        predicate_ids = list(report.predicates)
        lines += ["", "## By tag", "", *_header("Tag", *[f"`{p}`" for p in predicate_ids])]
        lines += [
            _row(tag, *[_slice_cell(row[p]) for p in predicate_ids]) for tag, row in report.slices.items()
        ]
    wrong = misses(report)
    lines += ["", f"## Misses ({len(wrong)})", ""]
    if wrong:
        lines += _header("Case", "Predicate", "Expected", "Got")
        lines += [
            _row(
                f"`{case_id}`",
                f"`{predicate_id}`",
                _expected_text(prediction.expected),
                _got_text(prediction),
            )
            for case_id, predicate_id, prediction in wrong
        ]
    else:
        lines.append("None.")
    usage = report.usage
    tokens = "n/a" if usage.input_tokens is None else f"{usage.input_tokens:,}"
    lines += [
        "",
        "## Usage",
        "",
        f"{usage.requests} requests, {tokens} input tokens, estimated ${fmt(usage.estimated_cost_usd, 4)}, "
        f"latency p50 {fmt(usage.latency_p50_ms, 0)} ms and p95 {fmt(usage.latency_p95_ms, 0)} ms; "
        f"{usage.replayed} of {usage.evaluations} evaluations replayed from cache.",
        "",
    ]
    return "\n".join(lines)


__all__ = [
    "CaseRecord",
    "CaseRun",
    "EvalReport",
    "Prediction",
    "SliceMetrics",
    "UsageSummary",
    "build_report",
    "misses",
    "render_markdown",
    "render_report",
]
