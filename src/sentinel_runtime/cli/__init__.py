"""The ``sentinel`` command line."""

from typing import Annotated

import typer

from sentinel_runtime import __version__
from sentinel_runtime.cli import evals as _evals
from sentinel_runtime.cli import predicates as _predicates
from sentinel_runtime.cli.bench import bench
from sentinel_runtime.cli.evaluate import evaluate
from sentinel_runtime.cli.models import models
from sentinel_runtime.cli.schema import schema

app = typer.Typer(
    name="sentinel",
    help="Sentinel: a semantic runtime. Evaluate semantic predicates and measure them.",
    no_args_is_help=True,
    add_completion=False,
    # Never print local variables in tracebacks: frames can hold request objects.
    pretty_exceptions_enable=False,
    pretty_exceptions_show_locals=False,
)


def _version(value: bool) -> None:
    if value:
        typer.echo(f"sentinel {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool, typer.Option("--version", callback=_version, is_eager=True, help="Show the version and exit.")
    ] = False,
) -> None:
    """Sentinel: a semantic runtime."""


app.command()(evaluate)
app.command()(schema)
app.command()(models)
app.command()(bench)
app.add_typer(_predicates.app, name="predicates")
app.add_typer(_evals.app, name="eval")

__all__ = ["app"]
