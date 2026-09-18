from datetime import UTC, datetime

import pytest

from sentinel_runtime import ChoiceResult, Evaluation, NoulResult, ProviderCall, ScoreResult, Usage


def _evaluation() -> Evaluation:
    return Evaluation(
        evaluation_id="ev_1",
        created_at=datetime(2026, 9, 18, tzinfo=UTC),
        predicate_set="demo",
        predicate_set_fingerprint="sha256:set",
        predicate_fingerprints={"urgent": "sha256:a", "team": "sha256:b", "tone": "sha256:c"},
        state_digest="sha256:state",
        state_chars=42,
        provider="typesafe",
        model_requested="jev-latest",
        model_resolved="jev-1.13.0",
        results={
            "urgent": NoulResult(probability=0.9, threshold=0.8, decision=True),
            "team": ChoiceResult(
                choice="billing",
                probabilities={"billing": 0.7, "tech": 0.3},
                confidence=0.4,
                min_confidence=0.6,
            ),
            "tone": ScoreResult(
                score=1.3,
                probabilities={0: 0.0, 1: 0.7, 2: 0.3},
                confidence=0.5,
                levels=3,
                normalized=0.65,
                nearest_level=1,
            ),
        },
        calls=(
            ProviderCall(
                request_id="req_1",
                model_requested="jev-latest",
                model_resolved="jev-1.13.0",
                input_tokens=300,
                output_tokens=40,
                estimated_input_tokens=360,
                latency_ms=120.0,
                predicate_ids=("urgent", "team", "tone"),
            ),
        ),
        usage=Usage(requests=1, input_tokens=300, output_tokens=40, estimated_cost_usd=0.0000126),
        latency_ms=125.0,
    )


def test_lookup_and_probability() -> None:
    evaluation = _evaluation()
    assert evaluation.probability("urgent") == 0.9
    assert isinstance(evaluation["team"], ChoiceResult)
    assert "tone" in evaluation
    with pytest.raises(TypeError, match="'team' is a choice result"):
        evaluation.probability("team")
    with pytest.raises(KeyError, match="no result for 'missing'; results: urgent, team, tone"):
        evaluation["missing"]


def test_json_round_trip_preserves_types() -> None:
    evaluation = _evaluation()
    restored = Evaluation.model_validate_json(evaluation.model_dump_json())
    assert restored == evaluation
    tone = restored["tone"]
    assert isinstance(tone, ScoreResult)
    assert tone.probabilities[1] == 0.7


def test_choice_top_probability() -> None:
    team = _evaluation()["team"]
    assert isinstance(team, ChoiceResult)
    assert team.top_probability == 0.7
    assert team.decision is None
