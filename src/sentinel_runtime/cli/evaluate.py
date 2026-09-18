"""``sentinel evaluate``: judge one state against a predicate set."""

from pathlib import Path
from typing import Annotated

import typer
from rich import box
from rich.markup import escape
from rich.table import Table

from sentinel_runtime.cli._common import (
    console,
    handle_errors,
    load_data_file,
    load_predicate_set,
    make_runtime,
    split_ids,
)
from sentinel_runtime.errors import ConfigError
from sentinel_runtime.results import ChoiceResult, Evaluation, NoulResult, Result
from sentinel_runtime.telemetry import JsonlSink, TelemetrySink


def _cells(result: Result) -> tuple[str, str, str]:
    """``(answer, detail, decision)`` cells for one result."""
    if isinstance(result, NoulResult):
        decision = "-"
        if result.threshold is not None:
            decision = f"{'yes' if result.decision else 'no'} (threshold {result.threshold:.2f})"
        return f"{result.probability:.2f}", "", decision
    if isinstance(result, ChoiceResult):
        detail = f"p {result.top_probability:.2f}, confidence {result.confidence:.2f}"
        decision = "-"
        if result.min_confidence is not None:
            verdict = "accept" if result.decision is not None else "abstain"
            decision = f"{verdict} (min confidence {result.min_confidence:.2f})"
        return result.choice, detail, decision
    detail = f"level {result.nearest_level} of 0-{result.levels - 1}, confidence {result.confidence:.2f}"
    return f"{result.score:.2f}", detail, "-"


def render_evaluation(evaluation: Evaluation) -> None:
    """Print an evaluation as a table followed by model and usage lines."""
    console.print(f"[bold]{escape(evaluation.predicate_set)}[/]  {evaluation.predicate_set_fingerprint}")
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    for column in ("predicate", "kind", "answer", "detail", "decision"):
        table.add_column(column, no_wrap=True)
    for predicate_id, result in evaluation.results.items():
        table.add_row(predicate_id, result.kind, *_cells(result))
    console.print(table)
    usage = evaluation.usage
    console.print(f"model {evaluation.model_resolved} (requested {evaluation.model_requested})", markup=False)
    parts = [f"{usage.requests} request{'s' if usage.requests != 1 else ''}"]
    if usage.input_tokens is not None:
        parts.append(f"{usage.input_tokens:,} input tokens")
    if usage.replayed:
        # A replay takes no time; the latency worth showing is the one measured when it was recorded.
        recorded = sum(call.latency_ms for call in evaluation.calls)
        parts.append(f"{recorded:.0f} ms when recorded")
    else:
        parts.append(f"{evaluation.latency_ms:.0f} ms")
    if usage.estimated_cost_usd is not None:
        parts.append(f"${usage.estimated_cost_usd:.6f}")
    if usage.replayed:
        parts.append("replayed from cache")
    console.print(", ".join(parts), markup=False)


def evaluate(
    state_file: Annotated[str, typer.Argument(help="State as a JSON or YAML file, or - for stdin.")],
    predicates: Annotated[str, typer.Option("--predicates", "-p", help="Predicate set path or name.")],
    select: Annotated[str | None, typer.Option(help="Comma-separated predicate ids to evaluate.")] = None,
    model: Annotated[str | None, typer.Option(help="Model to request, such as jev-1.13.0.")] = None,
    cache: Annotated[
        Path | None, typer.Option(help="Serve and record responses in this JSONL cache.")
    ] = None,
    offline: Annotated[bool, typer.Option(help="Use only the cache; fail on a miss. Needs --cache.")] = False,
    events: Annotated[
        Path | None, typer.Option(help="Append the evaluation event to this JSONL file.")
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print the evaluation as JSON.")] = False,
) -> None:
    """Evaluate a predicate set against one state."""
    with handle_errors():
        predicate_set = load_predicate_set(predicates)
        state = load_data_file(state_file, "state file")
        sinks: list[TelemetrySink] = [JsonlSink(events)] if events is not None else []
        ids = split_ids(select)
        if ids is not None:
            try:
                predicate_set = predicate_set.select(ids)
            except KeyError as error:
                raise ConfigError(str(error.args[0])) from None
        with make_runtime(model=model, cache=cache, offline=offline, sinks=sinks) as runtime:
            evaluation = runtime.evaluate(state, predicate_set)
        if json_output:
            console.print_json(evaluation.model_dump_json())
        else:
            render_evaluation(evaluation)
