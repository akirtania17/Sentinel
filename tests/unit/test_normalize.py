from typing import Any

import pytest

from sentinel_runtime import ChoiceResult, NoulResult, ProviderError, ScoreResult, choice, noul, score
from sentinel_runtime.normalize import normalize_answer, normalize_answers
from sentinel_runtime.providers import choice_answer, noul_answer, score_answer

NOUL = noul("urgent", question="Urgent?", threshold=0.7)
CHOICE = choice(
    "team", question="Which team?", options={"billing": None, "tech": None, "sales": None}, min_confidence=0.6
)
SCORE = score("tone", question="How angry?", levels=["calm", "annoyed", "furious"])


def test_noul_probability_and_decision() -> None:
    result = normalize_answer(NOUL, noul_answer(0.7), provider="p")
    assert result == NoulResult(probability=0.7, threshold=0.7, decision=True)
    below = normalize_answer(NOUL, noul_answer(0.69), provider="p")
    assert isinstance(below, NoulResult)
    assert below.decision is False
    no_threshold = normalize_answer(noul("x", question="q?"), noul_answer(0.9), provider="p")
    assert isinstance(no_threshold, NoulResult)
    assert no_threshold.decision is None


def test_float_artifacts_are_rounded() -> None:
    result = normalize_answer(NOUL, noul_answer(0.39999999999999997), provider="p")
    assert isinstance(result, NoulResult)
    assert result.probability == 0.4


def test_choice_orders_probabilities_by_option_and_applies_min_confidence() -> None:
    raw = choice_answer({"sales": 0.1, "tech": 0.2, "billing": 0.7}, confidence=0.65)
    result = normalize_answer(CHOICE, raw, provider="p")
    assert isinstance(result, ChoiceResult)
    assert list(result.probabilities) == ["billing", "tech", "sales"]
    assert result.choice == "billing"
    assert result.decision == "billing"
    abstain = normalize_answer(
        CHOICE, choice_answer({"billing": 0.4, "tech": 0.35, "sales": 0.25}, confidence=0.1), provider="p"
    )
    assert isinstance(abstain, ChoiceResult)
    assert abstain.decision is None


def test_choice_ties_accept_either_top_option() -> None:
    raw = choice_answer({"billing": 0.5, "tech": 0.5, "sales": 0.0}, choice="tech", confidence=0.4)
    result = normalize_answer(CHOICE, raw, provider="p")
    assert isinstance(result, ChoiceResult)
    assert result.choice == "tech"


def test_score_derived_fields() -> None:
    result = normalize_answer(SCORE, score_answer([0.0, 0.7, 0.3], confidence=0.54), provider="p")
    assert isinstance(result, ScoreResult)
    assert result.score == 1.3
    assert result.normalized == 0.65
    assert result.nearest_level == 1
    assert result.probabilities == {0: 0.0, 1: 0.7, 2: 0.3}
    top = normalize_answer(SCORE, score_answer([0.0, 0.0, 1.0]), provider="p")
    assert isinstance(top, ScoreResult)
    assert top.nearest_level == 2
    half = normalize_answer(SCORE, score_answer([0.25, 0.5, 0.25]), provider="p")
    assert isinstance(half, ScoreResult)
    assert half.nearest_level == 1


def test_sums_within_rounding_tolerance_are_accepted() -> None:
    raw = choice_answer({"billing": 0.34, "tech": 0.33, "sales": 0.32}, confidence=0.0)
    assert isinstance(normalize_answer(CHOICE, raw, provider="p"), ChoiceResult)


@pytest.mark.parametrize(
    ("predicate", "raw", "fragment"),
    [
        (NOUL, {"type": "choice", "noul": 0.5}, "does not match predicate kind"),
        (NOUL, {"type": "noul"}, "noul is not a number"),
        (NOUL, {"type": "noul", "noul": True}, "noul is not a number"),
        (NOUL, {"type": "noul", "noul": 1.2}, "outside [0, 1]"),
        (NOUL, {"type": "noul", "noul": float("nan")}, "outside [0, 1]"),
        (CHOICE, choice_answer({"billing": 0.7, "tech": 0.3}), "missing 'sales'"),
        (CHOICE, choice_answer({"billing": 0.6, "tech": 0.2, "sales": 0.1, "hr": 0.1}), "unexpected 'hr'"),
        (CHOICE, choice_answer({"billing": 0.6, "tech": 0.6, "sales": 0.1}), "sum to 1.3000"),
        (
            CHOICE,
            choice_answer({"billing": 0.2, "tech": 0.7, "sales": 0.1}, choice="billing"),
            "not the most probable",
        ),
        (
            CHOICE,
            choice_answer({"billing": 0.7, "tech": 0.2, "sales": 0.1}, choice="hr"),
            "not one of the predicate's options",
        ),
        (
            CHOICE,
            {**choice_answer({"billing": 1.0, "tech": 0.0, "sales": 0.0}), "confidence": None},
            "confidence is not a number",
        ),
        (
            SCORE,
            {"type": "score", "score": 1.0, "probabilities": {"0": 1.0}, "confidence": 1.0},
            "missing '1', '2'",
        ),
        (SCORE, {**score_answer([0.0, 0.0, 1.0]), "score": 2.5}, "outside [0, 2]"),
        (SCORE, {**score_answer([1.0, 0.0, 0.0]), "score": "low"}, "score is not a number"),
        (NOUL, "not an object", "answer is not an object"),
    ],
)
def test_malformed_answers_are_provider_errors(predicate: Any, raw: Any, fragment: str) -> None:
    with pytest.raises(ProviderError) as caught:
        normalize_answer(predicate, raw, provider="p")
    assert caught.value.kind == "malformed_response"
    assert fragment in caught.value.detail
    assert caught.value.detail.startswith(f"answer for {predicate.id!r}")


def test_missing_answers_are_reported_together() -> None:
    with pytest.raises(ProviderError, match="no answer for 'team', 'tone'"):
        normalize_answers((NOUL, CHOICE, SCORE), {"urgent": noul_answer(0.1)}, provider="p")


def test_extra_answers_are_ignored() -> None:
    results = normalize_answers(
        (NOUL,), {"urgent": noul_answer(0.1), "other": noul_answer(0.2)}, provider="p"
    )
    assert list(results) == ["urgent"]
