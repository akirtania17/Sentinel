"""``sentinel eval``: run fixture suites, compare reports, accept baselines, measure stability."""

import asyncio
import os
from pathlib import Path
from typing import Annotated

import typer
from rich import box
from rich.markup import escape
from rich.table import Table

from sentinel_runtime import __version__
from sentinel_runtime.cli._common import (
    EXIT_REGRESSION,
    console,
    err_console,
    handle_errors,
    split_ids,
)
from sentinel_runtime.config import SentinelConfig
from sentinel_runtime.errors import ConfigError, FixtureValidationError
from sentinel_runtime.evals.compare import DEFAULT_TOLERANCE, Comparison, compare_reports
from sentinel_runtime.evals.fixtures import EvalSuite
from sentinel_runtime.evals.report import (
    EvalReport,
    build_report,
    fmt,
    operating_thresholds,
    render_markdown,
    render_report,
)
from sentinel_runtime.evals.runner import DEFAULT_CONCURRENCY, run_cases
from sentinel_runtime.evals.stability import measure_stability, sample_cases
from sentinel_runtime.providers.caching import CacheMode, CachingProvider
from sentinel_runtime.providers.typesafe import TypeSafeProvider
from sentinel_runtime.runtime import Runtime

app = typer.Typer(help="Measure predicate quality against labeled fixtures.", no_args_is_help=True)

BASELINE_FILE = "baseline.json"
CACHE_FILE = "cache.jsonl"


def _has_key() -> bool:
    return bool(os.environ.get("TYPESAFE_API_KEY", "").strip())


def _thresholds(values: list[str]) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for value in values:
        name, separator, number = value.partition("=")
        try:
            threshold = float(number)
        except ValueError:
            threshold = -1.0
        if not separator or not name.strip() or not 0.0 <= threshold <= 1.0:
            raise ConfigError(f"--threshold expects id=value with a value from 0 to 1, got {value!r}")
        parsed[name.strip()] = threshold
    return parsed


def _mode(*, offline: bool, refresh: bool, cached: bool) -> CacheMode | None:
    if offline and refresh:
        raise ConfigError("--offline and --refresh cannot be combined")
    if not cached:
        if offline or refresh:
            raise ConfigError("--offline and --refresh need the response cache; drop --no-cache")
        return None
    if offline:
        return "offline"
    if refresh:
        return "record"
    return "replay" if _has_key() else "offline"


def _warn_on_model_drift(runtime: Runtime, model: str | None, requested: str) -> None:
    """Warn when a replayed cache was recorded with a model the requested alias no longer resolves to."""
    provider = runtime.provider
    if not isinstance(provider, CachingProvider) or provider.size == 0:
        return
    recorded = provider.models_resolved()
    if requested in recorded:
        return
    resolved = runtime.resolve_model(model)
    if resolved not in recorded:
        err_console.print(
            f"[yellow]warning:[/] cached responses were recorded with {escape(', '.join(sorted(recorded)))}, "
            f"but {escape(requested)} now resolves to {escape(resolved)}; run with --refresh to re-record"
        )


def render_comparison(comparison: Comparison) -> None:
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    for column in ("predicate", "metric", "baseline", "current", "change"):
        table.add_column(column, no_wrap=True)
    for change in comparison.metrics:
        delta = "n/a" if change.delta is None else f"{change.delta:+.2f}"
        table.add_row(
            change.predicate,
            change.metric.replace("_", " "),
            fmt(change.baseline),
            fmt(change.current),
            delta,
        )
    console.print(table)
    if comparison.model_upgrade:
        console.print(
            f"model: {', '.join(comparison.baseline_models)} -> {', '.join(comparison.current_models)}",
            markup=False,
        )
    for change in comparison.changes:
        state = "now correct" if change.now_correct else "now wrong"
        note = " (fragile: within repeat noise)" if change.fragile else ""
        console.print(f"  {change.case_id}  {change.predicate}: {state}{note}", markup=False)
    if comparison.shifts:
        console.print(
            f"{len(comparison.shifts)} probabilities moved by more than {comparison.tolerance:.2f}",
            markup=False,
        )
    if comparison.cases_added or comparison.cases_removed:
        console.print(
            f"cases added: {len(comparison.cases_added)}, removed: {len(comparison.cases_removed)}",
            markup=False,
        )
    if comparison.passed:
        console.print("[green]no regressions[/] against the baseline")
    else:
        console.print(f"[bold red]{len(comparison.regressions)} regression(s):[/]")
        for regression in comparison.regressions:
            console.print(f"  {escape(regression)}")


@app.command("run")
def run(
    suite: Annotated[str, typer.Argument(help="Suite name, directory, or cases file.")],
    model: Annotated[str | None, typer.Option(help="Model to request.")] = None,
    cache: Annotated[
        Path | None, typer.Option(help="Response cache. Default: cache.jsonl next to the cases.")
    ] = None,
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Call the provider for every case; record nothing.")
    ] = False,
    refresh: Annotated[bool, typer.Option(help="Re-record every response (needs TYPESAFE_API_KEY).")] = False,
    offline: Annotated[
        bool, typer.Option(help="Only replay the cache; fail listing any missing cases.")
    ] = False,
    prune: Annotated[
        bool, typer.Option(help="Drop cache entries this run did not use (stale questions, deleted cases).")
    ] = False,
    concurrency: Annotated[int, typer.Option(min=1, help="Cases evaluated at once.")] = DEFAULT_CONCURRENCY,
    tags: Annotated[
        str | None, typer.Option(help="Only cases with any of these comma-separated tags.")
    ] = None,
    exclude_tags: Annotated[str | None, typer.Option(help="Skip cases with any of these tags.")] = None,
    include_unreviewed: Annotated[
        bool, typer.Option(help="Also score cases marked reviewed: false.")
    ] = False,
    threshold: Annotated[
        list[str] | None, typer.Option(help="Operating threshold override, id=value. Repeatable.")
    ] = None,
    max_fpr: Annotated[
        float, typer.Option(min=0.0, max=1.0, help="False-positive ceiling for the recall recommendation.")
    ] = 0.05,
    baseline: Annotated[
        Path | None, typer.Option(help="Compare against this report. Default: baseline.json if present.")
    ] = None,
    fail_on_regression: Annotated[
        bool, typer.Option(help="Exit 3 when the comparison finds regressions.")
    ] = False,
    tolerance: Annotated[
        float, typer.Option(min=0.0, max=1.0, help="Regression tolerance.")
    ] = DEFAULT_TOLERANCE,
    allow_model_change: Annotated[
        bool, typer.Option(help="Compare against a baseline from a different model.")
    ] = False,
    out: Annotated[Path | None, typer.Option(help="Write the JSON report here.")] = None,
    markdown: Annotated[Path | None, typer.Option(help="Write a Markdown report here.")] = None,
    details: Annotated[
        bool, typer.Option(help="Also print threshold sweeps and confusion matrices.")
    ] = False,
    json_output: Annotated[
        bool, typer.Option("--json", help="Print the report as JSON instead of tables.")
    ] = False,
) -> None:
    """Evaluate a fixture suite and report predicate quality."""
    with handle_errors():
        config = SentinelConfig.load()
        loaded = EvalSuite.load(suite, config=config)
        wanted, unwanted = split_ids(tags), split_ids(exclude_tags)
        selected = loaded.select(tags=wanted, exclude_tags=unwanted, include_unreviewed=include_unreviewed)
        matching = loaded.select(tags=wanted, exclude_tags=unwanted, include_unreviewed=True)
        if not selected:
            raise FixtureValidationError("no cases match the selection", path=str(loaded.path))
        overrides = _thresholds(threshold or [])
        mode = _mode(offline=offline, refresh=refresh, cached=not no_cache)
        cache_path = cache if cache is not None else loaded.directory / CACHE_FILE
        runtime = Runtime.from_env(
            model=model,
            cache=None if mode is None else cache_path,
            cache_mode=mode or "replay",
            config=config,
        )
        with runtime:
            requested = runtime.model_for(model)
            if mode == "replay":
                _warn_on_model_drift(runtime, model, requested)
            runs = asyncio.run(run_cases(runtime, loaded.predicate_set, selected, concurrency=concurrency))
            if prune:
                cached = runtime.provider
                if not isinstance(cached, CachingProvider):
                    raise ConfigError("--prune needs the response cache; drop --no-cache")
                removed = cached.prune()
                err_console.print(f"pruned {removed} unused cache entr{'y' if removed == 1 else 'ies'}")
        report = build_report(
            loaded,
            runs,
            model_requested=requested,
            sentinel_version=__version__,
            threshold_overrides=overrides,
            max_fpr=max_fpr,
            tags=wanted,
            exclude_tags=unwanted,
            unreviewed_excluded=len(matching) - len(selected),
        )
        if out is not None:
            report.save(out)
        if markdown is not None:
            markdown.parent.mkdir(parents=True, exist_ok=True)
            markdown.write_text(render_markdown(report), encoding="utf-8")
        if json_output:
            console.print_json(report.model_dump_json())
        else:
            render_report(report, console, details=details)
        baseline_path = baseline if baseline is not None else loaded.directory / BASELINE_FILE
        comparison: Comparison | None = None
        if baseline is not None or (fail_on_regression and baseline_path.exists()):
            comparison = compare_reports(
                EvalReport.load(baseline_path),
                report,
                tolerance=tolerance,
                allow_model_change=allow_model_change,
            )
        elif fail_on_regression:
            raise ConfigError(f"--fail-on-regression needs a baseline; none at {baseline_path}")
    if comparison is not None:
        if not json_output:
            console.print(f"\n[bold]against {escape(str(baseline_path))}[/]")
            render_comparison(comparison)
        if fail_on_regression and not comparison.passed:
            raise typer.Exit(EXIT_REGRESSION)


@app.command("compare")
def compare(
    baseline: Annotated[Path, typer.Argument(help="The baseline report.")],
    current: Annotated[Path, typer.Argument(help="The report to check.")],
    tolerance: Annotated[
        float, typer.Option(min=0.0, max=1.0, help="Regression tolerance.")
    ] = DEFAULT_TOLERANCE,
    allow_model_change: Annotated[bool, typer.Option(help="Compare reports from different models.")] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Print the comparison as JSON.")] = False,
) -> None:
    """Compare two reports; exit 3 when the current one regresses."""
    with handle_errors():
        comparison = compare_reports(
            EvalReport.load(baseline),
            EvalReport.load(current),
            tolerance=tolerance,
            allow_model_change=allow_model_change,
        )
    if json_output:
        console.print_json(comparison.model_dump_json())
    else:
        render_comparison(comparison)
    if not comparison.passed:
        raise typer.Exit(EXIT_REGRESSION)


@app.command("accept")
def accept(
    report: Annotated[Path, typer.Argument(help="A reviewed report to make the suite's baseline.")],
    suite: Annotated[str, typer.Option(help="The suite the report belongs to.")],
    force: Annotated[bool, typer.Option(help="Accept even if some cases failed.")] = False,
) -> None:
    """Make a reviewed report the suite's regression baseline."""
    with handle_errors():
        loaded = EvalSuite.load(suite)
        accepted = EvalReport.load(report)
        if accepted.predicate_set_fingerprint != loaded.predicate_set.fingerprint:
            raise ConfigError(
                "the report was produced with a different predicate set than the suite now uses; "
                "run the suite again"
            )
        if accepted.suite != loaded.name:
            raise ConfigError(f"the report is for suite {accepted.suite!r}, not {loaded.name!r}")
        if accepted.cases_failed and not force:
            raise ConfigError(
                f"{accepted.cases_failed} case(s) failed in this report; fix them or pass --force"
            )
        target = loaded.directory / BASELINE_FILE
        accepted.save(target)
    console.print(f"baseline for {escape(loaded.name)} is now {escape(str(target))}")


@app.command("stability")
def stability(
    suite: Annotated[str, typer.Argument(help="Suite name, directory, or cases file.")],
    repeats: Annotated[int, typer.Option(min=2, help="Evaluations per case.")] = 5,
    sample: Annotated[int, typer.Option(min=1, help="Number of cases to sample.")] = 30,
    seed: Annotated[int, typer.Option(help="Sampling seed.")] = 0,
    model: Annotated[str | None, typer.Option(help="Model to request.")] = None,
    tags: Annotated[str | None, typer.Option(help="Only cases with any of these tags.")] = None,
    concurrency: Annotated[int, typer.Option(min=1)] = DEFAULT_CONCURRENCY,
    out: Annotated[Path | None, typer.Option(help="Write the JSON stability report here.")] = None,
) -> None:
    """Evaluate a sample repeatedly (live) and report how much answers move."""
    with handle_errors():
        config = SentinelConfig.load()
        loaded = EvalSuite.load(suite, config=config)
        cases = sample_cases(loaded.select(tags=split_ids(tags)), sample, seed)
        thresholds = {
            pid: value
            for pid, (value, _) in operating_thresholds(loaded.predicate_set.predicates, {}).items()
        }
        with Runtime(TypeSafeProvider(), model=model, config=config) as runtime:
            result = asyncio.run(
                measure_stability(
                    runtime,
                    loaded.name,
                    loaded.predicate_set,
                    cases,
                    repeats=repeats,
                    thresholds=thresholds,
                    seed=seed,
                    concurrency=concurrency,
                )
            )
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    console.print(
        f"[bold]{escape(result.suite)}[/]  {len(result.sample)} cases x {result.repeats} repeats, "
        f"model {escape(', '.join(result.models_resolved))}"
    )
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    for column in ("predicate", "kind", "mean spread", "max spread", "decision changed"):
        table.add_column(column, no_wrap=True)
    for predicate_id, entry in result.predicates.items():
        table.add_row(
            predicate_id,
            entry.kind,
            f"{entry.mean_spread:.3f}",
            f"{entry.max_spread:.3f}",
            f"{entry.changed} of {entry.cases}",
        )
    console.print(table)
    for answer in result.noisiest[:5]:
        console.print(
            f"  {answer.case_id}  {answer.predicate}: {answer.minimum:.2f} to {answer.maximum:.2f}",
            markup=False,
        )
