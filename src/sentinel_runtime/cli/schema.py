"""``sentinel schema``: JSON Schemas for Sentinel's file and event formats."""

import json
from enum import StrEnum
from typing import Annotated, Any

import typer

from sentinel_runtime.cli._common import console
from sentinel_runtime.predicates import PredicateFileV1
from sentinel_runtime.telemetry import EvaluationEvent


class SchemaName(StrEnum):
    predicates = "predicates"
    event = "event"


def _schema(name: SchemaName) -> dict[str, Any]:
    if name is SchemaName.predicates:
        return PredicateFileV1.model_json_schema()
    return EvaluationEvent.model_json_schema()


def schema(name: Annotated[SchemaName, typer.Argument(help="Which format.")]) -> None:
    """Print the JSON Schema for a Sentinel format."""
    console.print_json(json.dumps(_schema(name)))
