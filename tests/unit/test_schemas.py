"""The published JSON Schemas are valid, and the repository's own files match them."""

import json
from pathlib import Path
from typing import Any

import pytest

from sentinel_runtime._yaml import load_yaml
from sentinel_runtime.cli.schema import SchemaName
from tests.unit._schemas import assert_matches_schema, published_schema, schema_problems

TITLES = {
    "predicates": "Sentinel predicate file (v1)",
    "evals": "Sentinel evaluation fixtures (v1)",
    "eval-cases": "Sentinel evaluation cases (v1)",
    "report": "Sentinel evaluation report (v1)",
    "event": "Sentinel evaluation event (v1)",
    "evaluation": "Sentinel evaluation (v1)",
    "comparison": "Sentinel report comparison (v1)",
    "bench": "Sentinel batching benchmark (v1)",
    "listing": "Sentinel predicate listing (v1)",
}


def _load(path: Path) -> Any:
    # Round-trip through JSON: YAML may use non-string keys (such as `true:`) that JSON spells as strings.
    return json.loads(json.dumps(load_yaml(path.read_text(encoding="utf-8"))))


def test_every_schema_is_published_with_a_title() -> None:
    assert {name.value for name in SchemaName} == set(TITLES)
    for name, title in TITLES.items():
        assert published_schema(name)["title"] == title


def test_reference_predicate_files_match(repo_root: Path) -> None:
    files = sorted((repo_root / "predicates").glob("*.yaml"))
    assert len(files) == 3
    for path in files:
        assert_matches_schema("predicates", _load(path))


def test_reference_suites_and_baselines_match(repo_root: Path) -> None:
    suites = sorted(path for path in (repo_root / "evals").iterdir() if path.is_dir())
    assert len(suites) == 3
    for suite in suites:
        cases = _load(suite / "cases.yaml")
        assert_matches_schema("evals", cases)
        for included in cases.get("include", []):
            assert_matches_schema("eval-cases", _load(suite / included))
        assert_matches_schema("report", json.loads((suite / "baseline.json").read_text(encoding="utf-8")))


@pytest.mark.parametrize(
    ("name", "document", "fragment"),
    [
        ("predicates", {"format": "sentinel/predicates/v1", "name": "x", "predicates": {}}, "predicates"),
        ("evaluation", {"evaluation_id": "ev_1"}, "required property"),
        ("listing", {"name": "x", "fingerprint": "sha256:0", "predicates": [{"id": "a"}]}, "predicates/0"),
    ],
)
def test_schemas_reject_invalid_documents(name: str, document: Any, fragment: str) -> None:
    problems = schema_problems(name, document)
    assert problems
    assert any(fragment in problem for problem in problems), problems
