"""``sentinel models``: the provider's models and what aliases resolve to."""

from typing import Annotated

import typer
from rich import box
from rich.table import Table

from sentinel_runtime.cli._common import console, handle_errors
from sentinel_runtime.config import SentinelConfig
from sentinel_runtime.providers.typesafe import TypeSafeProvider
from sentinel_runtime.runtime import Runtime


def models(
    probe: Annotated[
        bool, typer.Option(help="Send one tiny request per alias to see the versioned model it resolves to.")
    ] = False,
) -> None:
    """List TypeSafe models and aliases available to your key."""
    with handle_errors(), Runtime(TypeSafeProvider(), config=SentinelConfig.load()) as runtime:
        provider = runtime.provider
        assert isinstance(provider, TypeSafeProvider)
        listing = provider.list_models()
        resolved = {entry["name"]: runtime.resolve_model(entry["name"]) for entry in listing} if probe else {}
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("name", no_wrap=True)
    if probe:
        table.add_column("resolves to", no_wrap=True)
    table.add_column("released")
    table.add_column("description")
    for entry in listing:
        cells = [entry["name"]]
        if probe:
            cells.append(resolved[entry["name"]])
        cells += [entry["release_date"][:10], entry["description"]]
        table.add_row(*cells)
    console.print(table)
