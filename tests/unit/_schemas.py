"""Validate documents against the JSON Schemas exactly as ``sentinel schema`` publishes them."""

import json
from functools import cache
from typing import Any

from jsonschema import Draft202012Validator
from typer.testing import CliRunner

from sentinel_runtime.cli import app


@cache
def published_schema(name: str) -> dict[str, Any]:
    result = CliRunner().invoke(app, ["schema", name])
    assert result.exit_code == 0, result.stderr
    schema: dict[str, Any] = json.loads(result.stdout)
    Draft202012Validator.check_schema(schema)
    return schema


def schema_problems(name: str, document: Any) -> list[str]:
    validator = Draft202012Validator(
        published_schema(name), format_checker=Draft202012Validator.FORMAT_CHECKER
    )
    return [
        f"{'/'.join(str(part) for part in error.absolute_path) or '<root>'}: {error.message}"
        for error in validator.iter_errors(document)
    ]


def assert_matches_schema(name: str, document: Any) -> None:
    problems = schema_problems(name, document)
    assert not problems, f"does not match the published `{name}` schema: " + "; ".join(problems[:10])
