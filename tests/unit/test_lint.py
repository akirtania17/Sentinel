from pathlib import Path

import pytest

from sentinel_runtime import PredicateSet, choice, noul, score
from sentinel_runtime.lint import lint_predicate, lint_predicate_set

STATE = {"user_request": "x", "proposed_action": {"tool": "t", "arguments": {"to": "a"}}, "steps": [{"n": 1}]}


def _codes(predicate: object, state: object = None) -> list[str]:
    return [warning.code for warning in lint_predicate(predicate, state=state)]  # type: ignore[arg-type]


def test_reference_sets_lint_clean(repo_root: Path) -> None:
    agent = PredicateSet.from_file(repo_root / "predicates" / "agent_action.yaml")
    assert lint_predicate_set(agent, state=STATE) == []


def test_state_references() -> None:
    assert _codes(noul("a", question="Does the action match?"), STATE) == ["no-state-reference"]
    assert (
        _codes(noul("a", question="Does `proposed_action.arguments.to` match `user_request`?"), STATE) == []
    )
    assert _codes(noul("a", question="Is `steps[0].n` set?"), STATE) == []
    assert _codes(noul("a", question="Does `proposed_actions` match?"), STATE) == ["unresolved-state-path"]
    assert _codes(noul("a", question="Is `steps[3]` set?"), STATE) == ["unresolved-state-path"]
    assert _codes(noul("a", question="Does the action match?")) == [], "no state, no reference checks"
    assert _codes(noul("a", question="Does it say `hello world`?"), STATE) == ["no-state-reference"]


@pytest.mark.parametrize(
    "question",
    [
        "How many items does `cart` contain?",
        "Is the total of `invoice` more than 100?",
        "What percentage of `rows` failed?",
    ],
)
def test_arithmetic(question: str) -> None:
    assert "arithmetic" in _codes(noul("a", question=question))


@pytest.mark.parametrize(
    "question",
    [
        "Was `order` placed before the deadline?",
        "Did it happen within 30 days of `signup`?",
        "Is `card` older than 2 years?",
    ],
)
def test_dates(question: str) -> None:
    assert "date-comparison" in _codes(noul("a", question=question))


@pytest.mark.parametrize(
    "question",
    [
        "Would a careful assistant ask before executing `proposed_action`?",
        "Does the action proceed without asking?",
        "What does `message` mean for the customer?",
        "Does this count as a refund request?",
        "Is `user_request` answered by `reply`?",
    ],
)
def test_ordinary_wording_is_not_flagged(question: str) -> None:
    assert _codes(noul("a", question=question)) == []


def test_negated_nouls() -> None:
    assert _codes(noul("a", question="Does `reply` not answer `question`?")) == ["negation"]
    assert _codes(noul("a", question="Doesn't `reply` answer `question`?")) == ["negation"]
    assert _codes(choice("a", question="Which step did not run?", options={"x": None, "y": None})) == []


def test_score_levels() -> None:
    assert _codes(score("a", question="How severe?", levels=["0", "1", "2"])) == ["vague-levels"]
    assert _codes(
        score("a", question="How severe?", levels=["Minor cosmetic issue", "Minor cosmetic issue"])
    ) == ["duplicate-levels"]
    assert (
        _codes(score("a", question="How severe?", levels=["No impact at all", "Blocks the whole team"])) == []
    )


def test_choice_escape_option() -> None:
    four = {"billing": None, "technical": None, "sales": None, "legal": None}
    assert _codes(choice("a", question="Which team?", options=four)) == ["no-escape-option"]
    assert _codes(choice("a", question="Which team?", options={**four, "other": None})) == []
    assert _codes(choice("a", question="Which team?", options={**four, "x": "None of the above"})) == []
    assert _codes(choice("a", question="Which team?", options={"not_applicable": None, **four})) == []
    assert _codes(choice("a", question="Which team?", options={"a": None, "b": None, "c": None})) == []


def test_long_and_structured_questions() -> None:
    assert _codes(noul("a", question="Is it? " + "x" * 700)) == ["long-question"]
    structured = noul("a", question={"question": "How many rows are there?", "focus": "`rows`"})
    assert "arithmetic" in _codes(structured)


def test_warnings_render_with_id_and_code() -> None:
    warning = lint_predicate(noul("urgent", question="How many times?"))[0]
    assert str(warning).startswith("urgent: [arithmetic]")


def test_negation_in_a_structured_question_checks_only_the_question_itself() -> None:
    examples = noul(
        "forbids",
        question={"question": "Does `user` forbid `action`?", "examples": ["do not send yet", "never email"]},
    )
    assert not [w for w in lint_predicate(examples) if w.code == "negation"]
    inverted = noul("inverted", question={"question": "Does `action` not follow from `user`?"})
    assert [w for w in lint_predicate(inverted) if w.code == "negation"]
