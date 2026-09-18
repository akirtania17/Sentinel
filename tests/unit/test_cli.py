"""The CLI through its real code paths, using an offline response cache instead of the network."""

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from sentinel_runtime import PredicateSet, Runtime
from sentinel_runtime.cli import app
from sentinel_runtime.config import SentinelConfig
from sentinel_runtime.providers import CachingProvider, FakeProvider, choice_answer, noul_answer, score_answer
from sentinel_runtime.telemetry import read_events
from tests.unit._schemas import assert_matches_schema

runner = CliRunner()
MODEL = "jev-1.13.0"
STATE = {
    "user_request": "Summarize my emails.",
    "proposed_action": {"tool": "send_email", "arguments": {"to": "a"}},
}
ANSWERS = {
    "action_matches_intent": noul_answer(0.02),
    "external_side_effect": noul_answer(0.95),
    "explicitly_authorized": noul_answer(0.01),
    "effect_class": choice_answer(
        {
            "read_only": 0.0,
            "local_change": 0.0,
            "external_communication": 1.0,
            "destructive_or_financial": 0.0,
            "other": 0.0,
        },
        confidence=1.0,
    ),
    "severity": score_answer([0.04, 0.4, 0.27, 0.14, 0.15], confidence=0.13),
}


@pytest.fixture
def predicates_path(repo_root: Path) -> Path:
    return repo_root / "predicates" / "agent_action.yaml"


@pytest.fixture
def state_file(tmp_path: Path) -> Path:
    path = tmp_path / "state.json"
    path.write_text(json.dumps(STATE))
    return path


@pytest.fixture
def cache(tmp_path: Path, predicates_path: Path) -> Path:
    """An offline cache holding one recorded evaluation of STATE."""
    path = tmp_path / "cache.jsonl"
    provider = CachingProvider(FakeProvider(ANSWERS, name="typesafe", model_resolved=MODEL), path)
    with Runtime(provider, model=MODEL, config=SentinelConfig()) as runtime:
        runtime.evaluate(STATE, PredicateSet.from_file(predicates_path))
    return path


def _evaluate(*args: str | Path) -> list[str]:
    return ["evaluate", *[str(arg) for arg in args]]


def test_version_and_help() -> None:
    assert runner.invoke(app, ["--version"]).stdout.startswith("sentinel ")
    result = runner.invoke(app, [])
    assert "evaluate" in result.stdout
    assert "predicates" in result.stdout


def test_evaluate_table_from_offline_cache(state_file: Path, predicates_path: Path, cache: Path) -> None:
    result = runner.invoke(
        app, _evaluate(state_file, "-p", predicates_path, "--model", MODEL, "--cache", cache, "--offline")
    )
    assert result.exit_code == 0, result.stderr
    out = result.stdout
    assert "agent_action  sha256:" in out
    assert "action_matches_intent" in out
    assert "no (threshold 0.50)" in out
    assert "external_communication" in out
    assert "accept (min confidence 0.60)" in out
    assert "level 2 of 0-4" in out
    assert "model jev-1.13.0 (requested jev-1.13.0)" in out
    assert "replayed from cache" in out
    assert re.search(r"\d+ ms when recorded", out)
    assert "…" not in out


def test_evaluate_json_and_events(
    tmp_path: Path, state_file: Path, predicates_path: Path, cache: Path
) -> None:
    events = tmp_path / "events.jsonl"
    result = runner.invoke(
        app,
        _evaluate(
            state_file,
            "-p",
            predicates_path,
            "--model",
            MODEL,
            "--cache",
            cache,
            "--offline",
            "--json",
            "--events",
            events,
        ),
    )
    assert result.exit_code == 0, result.stderr
    document = json.loads(result.stdout)
    assert document["model_resolved"] == MODEL
    assert document["results"]["action_matches_intent"] == {
        "kind": "noul",
        "probability": 0.02,
        "threshold": 0.5,
        "decision": False,
    }
    assert [event.outcome for event in read_events(events)] == ["ok"]
    assert_matches_schema("evaluation", document)
    for line in events.read_text().splitlines():
        assert_matches_schema("event", json.loads(line))


def test_evaluate_select(state_file: Path, predicates_path: Path, tmp_path: Path) -> None:
    path = tmp_path / "select.jsonl"
    provider = CachingProvider(FakeProvider(ANSWERS, name="typesafe", model_resolved=MODEL), path)
    with Runtime(provider, model=MODEL, config=SentinelConfig()) as runtime:
        runtime.evaluate(STATE, PredicateSet.from_file(predicates_path).select(["severity"]))
    result = runner.invoke(
        app,
        _evaluate(
            state_file,
            "-p",
            predicates_path,
            "--model",
            MODEL,
            "--cache",
            path,
            "--offline",
            "--select",
            "severity",
        ),
    )
    assert result.exit_code == 0, result.stderr
    assert "severity" in result.stdout
    assert "action_matches_intent" not in result.stdout
    unknown = runner.invoke(
        app, _evaluate(state_file, "-p", predicates_path, "--select", "nope", "--cache", path, "--offline")
    )
    assert unknown.exit_code == 1
    assert "no predicate 'nope'" in unknown.stderr


def test_cache_miss_exits_4(state_file: Path, predicates_path: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app, _evaluate(state_file, "-p", predicates_path, "--cache", tmp_path / "empty.jsonl", "--offline")
    )
    assert result.exit_code == 4
    assert "not in the cache" in result.stderr


def test_malformed_cached_answer_is_a_provider_error(
    tmp_path: Path, state_file: Path, predicates_path: Path
) -> None:
    path = tmp_path / "bad.jsonl"
    provider = CachingProvider(
        FakeProvider({**ANSWERS, "action_matches_intent": {"type": "noul", "noul": 3}}, name="typesafe"), path
    )
    runtime = Runtime(provider, model=MODEL, config=SentinelConfig())
    with pytest.raises(Exception, match="malformed_response"):
        runtime.evaluate(STATE, PredicateSet.from_file(predicates_path))
    runtime.close()
    result = runner.invoke(
        app, _evaluate(state_file, "-p", predicates_path, "--model", MODEL, "--cache", path, "--offline")
    )
    assert result.exit_code == 2
    assert "malformed_response" in result.stderr


@pytest.mark.parametrize(
    ("args", "fragment"),
    [
        (["missing.json", "-p", "agent_action"], "cannot read state file"),
        (["{state}", "-p", "no_such_set"], "predicate set 'no_such_set' not found"),
        (["{state}", "-p", "agent_action", "--offline"], "--offline and --refresh need --cache"),
    ],
)
def test_usage_errors_exit_1(args: list[str], fragment: str, state_file: Path) -> None:
    result = runner.invoke(app, ["evaluate", *[arg.replace("{state}", str(state_file)) for arg in args]])
    assert result.exit_code == 1
    assert fragment in result.stderr


def test_invalid_state_file(tmp_path: Path) -> None:
    path = tmp_path / "state.yaml"
    path.write_text("a: [unclosed\n")
    result = runner.invoke(app, ["evaluate", str(path), "-p", "agent_action"])
    assert result.exit_code == 1
    assert "not valid JSON or YAML" in result.stderr


def test_missing_key_exits_2(state_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    result = runner.invoke(app, ["evaluate", str(state_file), "-p", "agent_action"])
    assert result.exit_code == 2
    assert "TYPESAFE_API_KEY is not set" in result.stderr
    models = runner.invoke(app, ["models"])
    assert models.exit_code == 2


def test_predicates_validate(tmp_path: Path, predicates_path: Path, state_file: Path) -> None:
    ok = runner.invoke(app, ["predicates", "validate", str(predicates_path), "--state", str(state_file)])
    assert ok.exit_code == 0
    assert "agent_action: 5 predicates valid, 0 warnings  sha256:" in ok.stdout
    noisy = tmp_path / "noisy.yaml"
    noisy.write_text(
        "format: sentinel/predicates/v1\nname: noisy\npredicates:\n"
        "  count: {kind: noul, question: 'How many emails are there?'}\n"
    )
    warned = runner.invoke(app, ["predicates", "validate", str(noisy)])
    assert warned.exit_code == 0
    assert "[arithmetic]" in warned.stderr
    strict = runner.invoke(app, ["predicates", "validate", str(noisy), "--strict"])
    assert strict.exit_code == 1
    broken = tmp_path / "broken.yaml"
    broken.write_text("format: sentinel/predicates/v1\nname: b\npredicates:\n  a: {kind: noul}\n")
    invalid = runner.invoke(app, ["predicates", "validate", str(broken)])
    assert invalid.exit_code == 1
    assert "predicates.a.question" in invalid.stderr


def test_predicates_show(predicates_path: Path) -> None:
    table = runner.invoke(app, ["predicates", "show", str(predicates_path)])
    assert table.exit_code == 0
    assert "effect_class" in table.stdout
    assert "0.60" in table.stdout
    assert "0.50" in table.stdout
    document = json.loads(runner.invoke(app, ["predicates", "show", "agent_action", "--json"]).stdout)
    assert document["name"] == "agent_action"
    assert [row["id"] for row in document["predicates"]][:2] == [
        "action_matches_intent",
        "external_side_effect",
    ]
    assert document["predicates"][4]["default_threshold"] is None
    assert_matches_schema("listing", document)


@pytest.mark.parametrize(
    ("name", "title"),
    [("predicates", "Sentinel predicate file (v1)"), ("event", "Sentinel evaluation event (v1)")],
)
def test_schema(name: str, title: str) -> None:
    result = runner.invoke(app, ["schema", name])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["title"] == title


def test_error_messages_are_printed_verbatim(tmp_path: Path) -> None:
    missing = tmp_path / "state[1].json"
    result = runner.invoke(app, ["evaluate", str(missing), "-p", "agent_action"])
    assert result.exit_code == 1
    assert "state[1].json" in result.stderr
