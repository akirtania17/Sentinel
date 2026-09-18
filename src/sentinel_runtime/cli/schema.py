"""``sentinel schema``: JSON Schemas for Sentinel's file and event formats."""

import json
from enum import StrEnum
from typing import Annotated, Any

import typer

from sentinel_runtime.cli._common import console
from sentinel_runtime.evals.fixtures import EvalFileV1
from sentinel_runtime.evals.report import EvalReport
from sentinel_runtime.predicates import PredicateFileV1
from sentinel_runtime.telemetry import EvaluationEvent


class SchemaName(StrEnum):
    predicates = "predicates"
    evals = "evals"
    report = "report"
    event = "event"


def _schema(name: SchemaName) -> dict[str, Any]:
    models = {
        SchemaName.predicates: PredicateFileV1,
        SchemaName.evals: EvalFileV1,
        SchemaName.report: EvalReport,
        SchemaName.event: EvaluationEvent,
    }
    return models[name].model_json_schema()


def schema(name: Annotated[SchemaName, typer.Argument(help="Which format.")]) -> None:
    """Print the JSON Schema for a Sentinel format."""
    console.print_json(json.dumps(_schema(name)))
