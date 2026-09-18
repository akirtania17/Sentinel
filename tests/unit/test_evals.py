"""The evaluation harness end to end with a scripted provider."""

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from sentinel_runtime import FixtureValidationError, Runtime
from sentinel_runtime.cli import app
from sentinel_runtime.config import SentinelConfig
from sentinel_runtime.evals import (
    EvalReport,
    EvalSuite,
    build_report,
    compare_reports,
    render_markdown,
    run_cases,
)
from sentinel_runtime.evals.metrics import ChoiceMetrics, NoulMetrics, ScoreMetrics
from sentinel_runtime.evals.stability import measure_stability, sample_cases
from sentinel_runtime.providers import (
    CachingProvider,
    FakeProvider,
    ProviderRequest,
    choice_answer,
    noul_answer,
    score_answer,
)

PREDICATES = """format: sentinel/predicates/v1
name: triage
predicates:
  urgent:
    kind: noul
    question: Does `message` convey urgency?
    threshold: 0.5
  team:
    kind: choice
    question: Which team should handle `message`?
    options: {billing: Charges, tech: Bugs, other: Anything else}
    min_confidence: 0.6
  tone:
    kind: score
    question: How frustrated is the author of `message`?
    levels: [Calm and neutral, Frustrated but civil, Angry or hostile]
"""


def _case(
    case_id: str,
    p: float,
    team: str,
    tone: list[float],
    expect: dict[str, Any],
    tags: list[str],
    **extra: Any,
) -> dict[str, Any]:
    state = {"message": f"message {case_id}", "p": p, "team": team, "tone": tone}
    return {"id": case_id, "state": state, "expect": expect, "tags": tags, **extra}


CASES = [
    _case(
        "urgent_right",
        0.9,
        "billing",
        [0.0, 0.2, 0.8],
        {"urgent": True, "team": "billing", "tone": 2},
        ["positive"],
    ),
    _case(
        "calm_right", 0.1, "tech", [0.9, 0.1, 0.0], {"urgent": False, "team": "tech", "tone": 0}, ["negative"]
    ),
    _case(
        "urgent_missed",
        0.3,
        "tech",
        [0.1, 0.8, 0.1],
        {"urgent": True, "team": "billing", "tone": {"min": 1}},
        ["positive", "boundary"],
    ),
    _case(
        "false_alarm",
        0.8,
        "other",
        [0.0, 0.5, 0.5],
        {"urgent": False, "team": ["tech", "other"], "tone": {"max": 1}},
        ["negative"],
    ),
    _case(
        "near_threshold", 0.55, "billing", [0.2, 0.6, 0.2], {"urgent": True, "team": "billing"}, ["boundary"]
    ),
    _case(
        "ambiguous",
        0.5,
        "other",
        [0.3, 0.4, 0.3],
        {"urgent": None, "team": None, "tone": None},
        ["ambiguous"],
    ),
    _case("unlabeled_team", 0.2, "tech", [1.0, 0.0, 0.0], {"urgent": False}, ["negative"]),
    _case(
        "draft", 0.9, "tech", [1.0, 0.0, 0.0], {"urgent": True}, ["matrix"], source="matrix", reviewed=False
    ),
]


def respond(request: ProviderRequest) -> dict[str, Any]:
    state = request.state.value
    assert isinstance(state, dict)
    answers: dict[str, Any] = {
        "urgent": noul_answer(state["p"]),
        "team": choice_answer(
            {o: (0.8 if o == state["team"] else 0.1) for o in ("billing", "tech", "other")}, confidence=0.7
        ),
        "tone": score_answer(state["tone"]),
    }
    return {p.id: answers[p.id] for p in request.predicates}


@pytest.fixture
def suite_dir(tmp_path: Path) -> Path:
    (tmp_path / "predicates").mkdir()
    (tmp_path / "predicates" / "triage.yaml").write_text(PREDICATES)
    directory = tmp_path / "evals" / "triage"
    directory.mkdir(parents=True)
    document = {
        "format": "sentinel/evals/v1",
        "predicate_set": "../../predicates/triage.yaml",
        "cases": CASES,
    }
    (directory / "cases.yaml").write_text(json.dumps(document, indent=2))
    return directory


def _runtime(provider: FakeProvider | CachingProvider | None = None) -> Runtime:
    if provider is None:
        provider = FakeProvider(respond=respond, name="typesafe", model_resolved="jev-1.13.0")
    return Runtime(provider, model="jev-1.13.0", config=SentinelConfig())


def _report(suite: EvalSuite, **kwargs: Any) -> EvalReport:
    runtime = _runtime()
    runs = asyncio.run(run_cases(runtime, suite.predicate_set, suite.select(), concurrency=3))
    return build_report(suite, runs, model_requested="jev-1.13.0", sentinel_version="test", **kwargs)


# ---- fixtures -------------------------------------------------------------------------------------


def test_load_and_select(suite_dir: Path) -> None:
    suite = EvalSuite.load(suite_dir / "cases.yaml")
    assert suite.name == "triage"
    assert suite.predicate_set.name == "triage"
    assert len(suite.cases) == 8
    assert len(suite.select()) == 7, "unreviewed cases are excluded by default"
    assert len(suite.select(include_unreviewed=True)) == 8
    assert [c.id for c in suite.select(tags=["boundary"])] == ["urgent_missed", "near_threshold"]
    assert "urgent_missed" not in [c.id for c in suite.select(exclude_tags=["boundary"])]
    assert EvalSuite.load(suite_dir).path == suite_dir / "cases.yaml"


def test_suites_resolve_by_name(suite_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(suite_dir.parents[1])
    assert EvalSuite.load("triage", config=SentinelConfig()).name == "triage"


@pytest.mark.parametrize(
    ("case", "fragment"),
    [
        ({"id": "a", "state": {"m": 1}, "expect": {"nope": True}}, "not a predicate in 'triage'"),
        (
            {"id": "a", "state": {"m": 1}, "expect": {"urgent": 1}},
            "noul expectation must be true, false, or null",
        ),
        ({"id": "a", "state": {"m": 1}, "expect": {"team": "sales"}}, "unknown option(s) 'sales'"),
        ({"id": "a", "state": {"m": 1}, "expect": {"team": []}}, "non-empty list of options"),
        ({"id": "a", "state": {"m": 1}, "expect": {"tone": 5}}, "score levels run from 0 to 2"),
        ({"id": "a", "state": {"m": 1}, "expect": {"tone": {"min": 3}}}, "score levels run from 0 to 2"),
        ({"id": "a", "state": {"m": 1}, "expect": {"tone": True}}, "a score expectation must be a level"),
        ({"id": "a", "state": "", "expect": {}}, "state is empty"),
        ({"id": "Bad-Id", "state": "x", "expect": {}}, "cases.0.id"),
        ({"id": "a", "state": "x", "expect": {}, "reviewd": True}, "Extra inputs"),
    ],
)
def test_fixture_errors(suite_dir: Path, case: dict[str, Any], fragment: str) -> None:
    path = suite_dir / "bad.yaml"
    document = {
        "format": "sentinel/evals/v1",
        "predicate_set": "../../predicates/triage.yaml",
        "cases": [case],
    }
    path.write_text(json.dumps(document))
    with pytest.raises(FixtureValidationError) as caught:
        EvalSuite.from_file(path, config=SentinelConfig())
    assert fragment in str(caught.value)


def test_duplicate_case_ids(suite_dir: Path) -> None:
    path = suite_dir / "dup.yaml"
    case: dict[str, Any] = {"id": "a", "state": "x", "expect": {}}
    path.write_text(
        json.dumps(
            {
                "format": "sentinel/evals/v1",
                "predicate_set": "../../predicates/triage.yaml",
                "cases": [case, case],
            }
        )
    )
    with pytest.raises(FixtureValidationError, match="duplicate case id"):
        EvalSuite.from_file(path, config=SentinelConfig())


def test_missing_predicate_set(suite_dir: Path) -> None:
    path = suite_dir / "orphan.yaml"
    path.write_text(
        json.dumps(
            {
                "format": "sentinel/evals/v1",
                "predicate_set": "missing.yaml",
                "cases": [{"id": "a", "state": "x", "expect": {}}],
            }
        )
    )
    with pytest.raises(FixtureValidationError, match="predicate file not found"):
        EvalSuite.from_file(path, config=SentinelConfig())


# ---- reports --------------------------------------------------------------------------------------


def test_report_metrics_and_records(suite_dir: Path) -> None:
    suite = EvalSuite.load(suite_dir)
    report = _report(suite, unreviewed_excluded=1)
    assert report.cases_selected == 7
    assert report.cases_evaluated == 7
    assert report.models_resolved == ["jev-1.13.0"]
    urgent = report.predicates["urgent"]
    assert isinstance(urgent, NoulMetrics)
    assert (urgent.positives, urgent.negatives, urgent.ambiguous) == (3, 3, 1)
    assert (urgent.at_threshold.tp, urgent.at_threshold.fp, urgent.at_threshold.fn) == (2, 1, 1)
    assert "near_threshold" in urgent.fragile
    team = report.predicates["team"]
    assert isinstance(team, ChoiceMetrics)
    assert team.labeled == 5
    assert team.accuracy == pytest.approx(4 / 5)
    tone = report.predicates["tone"]
    assert isinstance(tone, ScoreMetrics)
    assert (tone.exact_labeled, tone.range_labeled) == (2, 2)
    records = {record.id: record for record in report.cases}
    assert records["urgent_missed"].predictions["urgent"].correct is False
    assert records["ambiguous"].predictions["urgent"].correct is None
    assert records["unlabeled_team"].predictions["team"].labeled is False
    assert report.slices["boundary"]["urgent"].labeled == 2
    assert report.usage.requests == 7


def test_threshold_overrides(suite_dir: Path) -> None:
    suite = EvalSuite.load(suite_dir)
    report = _report(suite, threshold_overrides={"urgent": 0.25})
    urgent = report.predicates["urgent"]
    assert isinstance(urgent, NoulMetrics)
    assert urgent.threshold_source == "override"
    assert urgent.at_threshold.fn == 0
    with pytest.raises(FixtureValidationError, match="non-noul or unknown"):
        _report(suite, threshold_overrides={"team": 0.5})


def test_report_round_trip_and_markdown(suite_dir: Path, tmp_path: Path) -> None:
    report = _report(EvalSuite.load(suite_dir))
    path = tmp_path / "report.json"
    report.save(path)
    assert EvalReport.load(path) == report
    markdown = render_markdown(report)
    for heading in (
        "# Evaluation: triage",
        "## Summary",
        "## `urgent`",
        "| Threshold | TP |",
        "## By tag",
        "## Misses (",
        "## Usage",
    ):
        assert heading in markdown
    assert "`urgent_missed`" in markdown
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    with pytest.raises(FixtureValidationError, match="not a Sentinel evaluation report"):
        EvalReport.load(bad)


# ---- comparison -----------------------------------------------------------------------------------


def test_identical_reports_pass(suite_dir: Path) -> None:
    report = _report(EvalSuite.load(suite_dir))
    comparison = compare_reports(report, report)
    assert comparison.passed
    assert comparison.changes == []


def _with_prediction(report: EvalReport, case_id: str, predicate: str, **changes: Any) -> EvalReport:
    data = report.model_dump(mode="json")
    for record in data["cases"]:
        if record["id"] == case_id:
            record["predictions"][predicate].update(changes)
    return EvalReport.model_validate(data)


def test_flips_are_regressions_unless_fragile(suite_dir: Path) -> None:
    baseline = _report(EvalSuite.load(suite_dir))
    solid = _with_prediction(baseline, "urgent_right", "urgent", value=0.2, correct=False)
    comparison = compare_reports(baseline, solid)
    assert not comparison.passed
    assert "urgent_right: urgent was correct and is now wrong" in comparison.regressions
    assert comparison.shifts[0].delta == pytest.approx(-0.7)
    fragile = _with_prediction(baseline, "near_threshold", "urgent", value=0.48, correct=False)
    flipped = compare_reports(baseline, fragile)
    assert flipped.passed
    assert flipped.changes[0].fragile


def test_metric_drops_fingerprints_and_models(suite_dir: Path) -> None:
    baseline = _report(EvalSuite.load(suite_dir))
    data = baseline.model_dump(mode="json")
    data["predicates"]["team"]["accuracy"] = 0.5
    data["predicate_set_fingerprint"] = "sha256:other"
    data["models_resolved"] = ["jev-9.0.0"]
    current = EvalReport.model_validate(data)
    regressions = " | ".join(compare_reports(baseline, current).regressions)
    assert "team accuracy dropped from 0.80 to 0.50" in regressions
    assert "predicate set changed" in regressions
    assert "model changed from jev-1.13.0 to jev-9.0.0" in regressions
    upgrade = compare_reports(baseline, current, allow_model_change=True)
    assert upgrade.model_upgrade
    assert not any("model changed" in r for r in upgrade.regressions)


def test_metric_becoming_undefined_is_a_regression(suite_dir: Path) -> None:
    baseline = _report(EvalSuite.load(suite_dir))
    data = baseline.model_dump(mode="json")
    data["predicates"]["team"]["accuracy"] = None
    assert any(
        "became undefined" in r
        for r in compare_reports(baseline, EvalReport.model_validate(data)).regressions
    )


# ---- stability ------------------------------------------------------------------------------------


def test_stability_measures_spread_and_changes(suite_dir: Path) -> None:
    suite = EvalSuite.load(suite_dir)
    calls = {"n": 0}

    def wobble(request: ProviderRequest) -> dict[str, Any]:
        calls["n"] += 1
        answers = respond(request)
        state = request.state.value
        assert isinstance(state, dict)
        if state["message"].endswith("near_threshold"):
            answers["urgent"] = noul_answer(0.55 if calls["n"] % 2 else 0.45)
        return answers

    runtime = _runtime(FakeProvider(respond=wobble, name="typesafe", model_resolved="jev-1.13.0"))
    cases = sample_cases(suite.select(), 4, seed=1)
    assert len(cases) == 4
    assert sample_cases(suite.select(), 4, seed=1) == cases
    near = [c for c in suite.select() if c.id == "near_threshold"]
    report = asyncio.run(
        measure_stability(
            runtime, "triage", suite.predicate_set, near, repeats=4, thresholds={"urgent": 0.5}, seed=0
        )
    )
    assert report.predicates["urgent"].changed == 1
    assert report.predicates["urgent"].max_spread == pytest.approx(0.1)
    assert report.predicates["team"].changed == 0
    assert report.noisiest[0].case_id == "near_threshold"


# ---- CLI ------------------------------------------------------------------------------------------

runner = CliRunner()


def _record_cache(suite_dir: Path) -> None:
    suite = EvalSuite.load(suite_dir)
    provider = CachingProvider(
        FakeProvider(respond=respond, name="typesafe", model_resolved="jev-1.13.0"), suite_dir / "cache.jsonl"
    )
    with _runtime(provider) as runtime:
        asyncio.run(run_cases(runtime, suite.predicate_set, suite.select(include_unreviewed=True)))


def test_cli_run_accept_and_gate(suite_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    _record_cache(suite_dir)
    out, markdown = tmp_path / "r.json", tmp_path / "r.md"
    result = runner.invoke(
        app,
        [
            "eval",
            "run",
            str(suite_dir),
            "--model",
            "jev-1.13.0",
            "--out",
            str(out),
            "--markdown",
            str(markdown),
        ],
    )
    assert result.exit_code == 0, result.stderr
    assert "7 of 7 cases evaluated, 1 unreviewed excluded" in result.stdout
    assert "Thresholds are empirical" in result.stdout
    assert "misses" in result.stdout
    assert "7 of 7 replayed from cache" in result.stdout
    assert markdown.read_text().startswith("# Evaluation: triage")

    gate = runner.invoke(
        app, ["eval", "run", str(suite_dir), "--model", "jev-1.13.0", "--fail-on-regression"]
    )
    assert gate.exit_code == 1
    assert "needs a baseline" in gate.stderr

    accepted = runner.invoke(app, ["eval", "accept", str(out), "--suite", str(suite_dir)])
    assert accepted.exit_code == 0, accepted.stderr
    assert (suite_dir / "baseline.json").exists()
    passing = runner.invoke(
        app, ["eval", "run", str(suite_dir), "--model", "jev-1.13.0", "--fail-on-regression"]
    )
    assert passing.exit_code == 0, passing.stderr
    assert "no regressions" in passing.stdout

    regressed = _with_prediction(EvalReport.load(out), "calm_right", "urgent", value=0.9, correct=False)
    regressed_path = tmp_path / "regressed.json"
    regressed.save(regressed_path)
    compared = runner.invoke(app, ["eval", "compare", str(out), str(regressed_path)])
    assert compared.exit_code == 3
    assert "calm_right: urgent was correct and is now wrong" in compared.stdout


def test_cli_offline_miss_lists_cases(suite_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    result = runner.invoke(app, ["eval", "run", str(suite_dir), "--model", "jev-1.13.0"])
    assert result.exit_code == 4
    assert "7 evaluation(s) are not in the cache (urgent_right, calm_right" in result.stderr


def test_cli_accept_rejects_other_predicate_sets(suite_dir: Path, tmp_path: Path) -> None:
    report = _report(EvalSuite.load(suite_dir))
    data = report.model_dump(mode="json")
    data["predicate_set_fingerprint"] = "sha256:stale"
    stale = tmp_path / "stale.json"
    stale.write_text(json.dumps(data))
    result = runner.invoke(app, ["eval", "accept", str(stale), "--suite", str(suite_dir)])
    assert result.exit_code == 1
    assert "different predicate set" in result.stderr


def test_cli_argument_errors(suite_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    bad = runner.invoke(app, ["eval", "run", str(suite_dir), "--threshold", "urgent"])
    assert bad.exit_code == 1
    assert "--threshold expects id=value" in bad.stderr
    both = runner.invoke(app, ["eval", "run", str(suite_dir), "--offline", "--refresh"])
    assert both.exit_code == 1
    none = runner.invoke(app, ["eval", "run", str(suite_dir), "--tags", "nothing_has_this"])
    assert none.exit_code == 1
    assert "no cases match" in none.stderr


def test_cli_json_output(suite_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    _record_cache(suite_dir)
    result = runner.invoke(app, ["eval", "run", str(suite_dir), "--model", "jev-1.13.0", "--json"])
    assert result.exit_code == 0, result.stderr
    assert json.loads(result.stdout)["format"] == "sentinel/eval-report/v1"


@pytest.mark.parametrize("name", ["evals", "report"])
def test_eval_schemas(name: str) -> None:
    assert runner.invoke(app, ["schema", name]).exit_code == 0
