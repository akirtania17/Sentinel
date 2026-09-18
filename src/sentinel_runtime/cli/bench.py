"""``sentinel bench``: measure what batching saves on a real state."""

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich import box
from rich.markup import escape
from rich.table import Table

from sentinel_runtime.bench import BenchReport, run_benchmark
from sentinel_runtime.cli._common import console, handle_errors, load_data_file, load_predicate_set
from sentinel_runtime.config import SentinelConfig
from sentinel_runtime.providers.typesafe import TypeSafeProvider
from sentinel_runtime.runtime import Runtime


def render_bench(report: BenchReport) -> None:
    console.print(
        f"[bold]{escape(report.predicate_set)}[/]  {report.predicates} predicates, "
        f"{report.state_chars} characters of state, model {escape(report.model_resolved)}, "
        f"{report.repeats} repeats"
    )
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    for column in ("strategy", "requests", "median latency", "input tokens", "estimated cost"):
        table.add_column(column, no_wrap=True, justify="left" if column == "strategy" else "right")
    for strategy in report.strategies:
        table.add_row(
            strategy.strategy,
            str(strategy.requests),
            f"{strategy.median_latency_ms:.0f} ms",
            "n/a" if strategy.input_tokens is None else f"{strategy.input_tokens:,}",
            "n/a" if strategy.estimated_cost_usd is None else f"${strategy.estimated_cost_usd:.6f}",
        )
    console.print(table)
    if report.cost_ratio is not None:
        console.print(
            f"one request per predicate costs {report.cost_ratio:.1f}x the batched request", markup=False
        )
    if report.latency_ratio is not None:
        console.print(
            f"sequential single requests take {report.latency_ratio:.1f}x as long as one batched request",
            markup=False,
        )
    parts: list[str] = []
    for kind in ("noul", "choice", "score"):
        deltas = [answer.delta for answer in report.answers if answer.kind == kind]
        if not deltas:
            continue
        numeric = [delta for delta in deltas if delta is not None]
        text = f"{kind}s {max(numeric):.3f}" if numeric else f"{kind}s n/a"
        changed = len(deltas) - len(numeric)
        if changed:
            text += f" ({changed} picked a different option)"
        parts.append(text)
    console.print(f"largest batched vs single answer difference: {', '.join(parts)}", markup=False)
    console.print(
        f"throughput: {report.throughput_evaluations_per_second:.1f} batched evaluations per second at "
        f"concurrency {report.concurrency}",
        markup=False,
    )


def bench(
    predicates: Annotated[str, typer.Argument(help="Predicate set path or name.")],
    state_file: Annotated[str, typer.Argument(help="State as a JSON or YAML file, or - for stdin.")],
    model: Annotated[str | None, typer.Option(help="Model to request.")] = None,
    repeats: Annotated[int, typer.Option(min=1, help="Measurements per strategy.")] = 3,
    concurrency: Annotated[
        int, typer.Option(min=1, help="Parallel requests for the concurrent strategy.")
    ] = 4,
    out: Annotated[Path | None, typer.Option(help="Write the JSON report here.")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print the report as JSON.")] = False,
) -> None:
    """Compare one batched request with one request per predicate (live)."""
    with handle_errors():
        predicate_set = load_predicate_set(predicates)
        state = load_data_file(state_file, "state file")
        with Runtime(TypeSafeProvider(), model=model, config=SentinelConfig.load()) as runtime:
            report = asyncio.run(
                run_benchmark(
                    runtime, state, predicate_set, repeats=repeats, concurrency=concurrency, model=model
                )
            )
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    if json_output:
        console.print_json(report.model_dump_json())
    else:
        render_bench(report)
