"""``sentinel predicates``: validate and inspect predicate sets."""

from typing import Annotated

import typer
from rich import box
from rich.markup import escape
from rich.table import Table

from sentinel_runtime.cli._common import (
    EXIT_USAGE,
    console,
    err_console,
    handle_errors,
    load_data_file,
    load_predicate_set,
)
from sentinel_runtime.lint import lint_predicate_set
from sentinel_runtime.predicates import PredicateListing

app = typer.Typer(help="Validate and inspect predicate sets.", no_args_is_help=True)


@app.command()
def validate(
    predicates: Annotated[str, typer.Argument(help="Predicate set path or name.")],
    state: Annotated[
        str | None, typer.Option(help="An example state to check backticked paths against.")
    ] = None,
    strict: Annotated[bool, typer.Option(help="Exit with an error when there are warnings.")] = False,
) -> None:
    """Check a predicate file and report authoring warnings."""
    with handle_errors():
        predicate_set = load_predicate_set(predicates)
        example = load_data_file(state, "state file") if state is not None else None
        warnings = lint_predicate_set(predicate_set, state=example)
    for warning in warnings:
        err_console.print(f"[yellow]warning:[/] {escape(str(warning))}")
    count, problems = len(predicate_set), len(warnings)
    console.print(
        f"{escape(predicate_set.name)}: {count} predicate{'s' if count != 1 else ''} valid, "
        f"{problems} warning{'s' if problems != 1 else ''}  {predicate_set.fingerprint}"
    )
    if strict and warnings:
        raise typer.Exit(EXIT_USAGE)


@app.command()
def show(
    predicates: Annotated[str, typer.Argument(help="Predicate set path or name.")],
    json_output: Annotated[bool, typer.Option("--json", help="Print as JSON.")] = False,
) -> None:
    """List a set's predicates with their kinds, fingerprints and default thresholds."""
    with handle_errors():
        predicate_set = load_predicate_set(predicates)
    listing = PredicateListing.of(predicate_set)
    if json_output:
        console.print_json(listing.model_dump_json())
        return
    console.print(f"[bold]{escape(predicate_set.name)}[/]  {predicate_set.fingerprint}")
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    for column in ("id", "kind", "fingerprint", "default", "tags"):
        table.add_column(column, no_wrap=True)
    for row in listing.predicates:
        default = row.default_threshold
        table.add_row(
            row.id,
            row.kind,
            row.fingerprint[:23],
            "-" if default is None else f"{default:.2f}",
            ", ".join(row.tags),
        )
    console.print(table)
