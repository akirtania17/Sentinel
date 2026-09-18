"""``sentinel schema``: JSON Schemas for Sentinel's file formats and command output."""

import json
from enum import StrEnum
from typing import Annotated, Any, Literal

import typer
from pydantic import BaseModel

from sentinel_runtime.bench import BenchReport
from sentinel_runtime.cli._common import console
from sentinel_runtime.evals.compare import Comparison
from sentinel_runtime.evals.fixtures import CaseFileV1, EvalFileV1
from sentinel_runtime.evals.report import EvalReport
from sentinel_runtime.predicates import PredicateFileV1, PredicateListing
from sentinel_runtime.results import Evaluation
from sentinel_runtime.telemetry import EvaluationEvent


class SchemaName(StrEnum):
    predicates = "predicates"
    evals = "evals"
    eval_cases = "eval-cases"
    report = "report"
    event = "event"
    evaluation = "evaluation"
    comparison = "comparison"
    bench = "bench"
    listing = "listing"


_Mode = Literal["validation", "serialization"]

# Files people write are described as Sentinel reads them; everything Sentinel writes (reports,
# events, and each command's --json output) is described exactly as it is written.
_SCHEMAS: dict[SchemaName, tuple[type[BaseModel], _Mode]] = {
    SchemaName.predicates: (PredicateFileV1, "validation"),
    SchemaName.evals: (EvalFileV1, "validation"),
    SchemaName.eval_cases: (CaseFileV1, "validation"),
    SchemaName.report: (EvalReport, "serialization"),
    SchemaName.event: (EvaluationEvent, "serialization"),
    SchemaName.evaluation: (Evaluation, "serialization"),
    SchemaName.comparison: (Comparison, "serialization"),
    SchemaName.bench: (BenchReport, "serialization"),
    SchemaName.listing: (PredicateListing, "serialization"),
}


def json_schema(name: SchemaName | str) -> dict[str, Any]:
    """The published JSON Schema for a file format or a command's JSON output."""
    model, mode = _SCHEMAS[SchemaName(name)]
    return model.model_json_schema(mode=mode)


def schema(
    name: Annotated[
        SchemaName,
        typer.Argument(
            help=(
                "Files you write: predicates, evals, eval-cases. Output: report (eval run --json and "
                "--out), event (--events files), evaluation (evaluate --json), comparison (eval compare "
                "--json), bench (bench --json), listing (predicates show --json)."
            )
        ),
    ],
) -> None:
    """Print the JSON Schema for a Sentinel file format or command output."""
    console.print_json(json.dumps(json_schema(name)))
