import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from sentinel_runtime import (
    ChoicePredicate,
    NoulPredicate,
    PredicateSet,
    PredicateValidationError,
    ScorePredicate,
    choice,
    noul,
    score,
)
from sentinel_runtime.predicates import PredicateFileV1


def _set() -> PredicateSet:
    return PredicateSet(
        name="demo",
        predicates=(
            noul("is_urgent", question="Does `message` convey urgency?", threshold=0.7),
            choice(
                "team",
                question="Which team should handle `message`?",
                options={"billing": "Charges", "tech": None},
            ),
            score(
                "tone",
                question="How frustrated is the author of `message`?",
                levels=["Calm", "Annoyed", "Angry"],
            ),
        ),
    )


# ---- construction and validation ----------------------------------------------------------------


def test_constructors_build_typed_predicates() -> None:
    predicate_set = _set()
    assert isinstance(predicate_set["is_urgent"], NoulPredicate)
    assert isinstance(predicate_set["team"], ChoicePredicate)
    assert isinstance(predicate_set["tone"], ScorePredicate)
    assert predicate_set.ids() == ("is_urgent", "team", "tone")
    assert len(predicate_set) == 3
    assert "team" in predicate_set
    assert "missing" not in predicate_set


@pytest.mark.parametrize("bad_id", ["", "Upper", "has-dash", "1starts_with_digit", "a" * 65, "sp ace"])
def test_rejects_non_identifier_ids(bad_id: str) -> None:
    with pytest.raises(ValidationError):
        noul(bad_id, question="q?")


@pytest.mark.parametrize("question", ["", "   ", {}, []])
def test_rejects_empty_questions(question: object) -> None:
    with pytest.raises(ValidationError, match="must not be empty"):
        noul("a", question=question)  # type: ignore[arg-type]


def test_structured_question_must_be_json() -> None:
    with pytest.raises(ValidationError, match="JSON-representable"):
        noul("a", question={"when": {1, 2}})


def test_choice_needs_two_to_255_options() -> None:
    with pytest.raises(ValidationError, match="at least 2"):
        choice("a", question="q?", options={"only": None})
    with pytest.raises(ValidationError, match="at most 255"):
        choice("a", question="q?", options={f"o{i}": None for i in range(256)})
    assert len(choice("a", question="q?", options={f"o{i}": None for i in range(255)}).options) == 255


def test_choice_option_keys_are_trimmed_and_non_empty() -> None:
    with pytest.raises(ValidationError, match="leading or trailing whitespace"):
        choice("a", question="q?", options={" x": None, "y": None})
    with pytest.raises(ValidationError):
        choice("a", question="q?", options={"": None, "y": None})


def test_score_needs_two_to_ten_levels() -> None:
    with pytest.raises(ValidationError, match="at least 2"):
        score("a", question="q?", levels=["one"])
    with pytest.raises(ValidationError, match="at most 10"):
        score("a", question="q?", levels=[f"level {i}" for i in range(11)])
    with pytest.raises(ValidationError, match="level 1 must not be empty"):
        score("a", question="q?", levels=["fine", " "])


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_threshold_must_be_a_probability(threshold: float) -> None:
    with pytest.raises(ValidationError):
        noul("a", question="q?", threshold=threshold)


def test_noul_criteria_must_describe_something() -> None:
    with pytest.raises(ValidationError, match="criteria must describe"):
        noul("a", question="q?", criteria={})
    only_true = noul("a", question="q?", criteria={"true": "it is"})
    assert only_true.criteria is not None
    assert only_true.criteria.wire() == {"true": "it is"}


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        NoulPredicate.model_validate({"id": "a", "question": "q?", "treshold": 0.5})


def test_predicate_set_rejects_duplicate_ids_and_empty_sets() -> None:
    with pytest.raises(ValidationError, match="duplicate predicate id"):
        PredicateSet(name="s", predicates=(noul("a", question="q?"), noul("a", question="r?")))
    with pytest.raises(ValidationError):
        PredicateSet(name="s", predicates=())


def test_predicates_are_immutable_and_hashable() -> None:
    predicate = choice("a", question={"q": "which?"}, options={"x": None, "y": None})
    with pytest.raises(ValidationError):
        predicate.id = "b"  # type: ignore[misc]
    same = choice("a", question={"q": "which?"}, options={"x": None, "y": None})
    assert predicate == same
    assert hash(predicate) == hash(same)
    assert len({predicate, same, noul("a", question="q?")}) == 2
    assert hash(_set()) == hash(_set())


# ---- fingerprints -------------------------------------------------------------------------------


def test_fingerprint_is_a_golden_value() -> None:
    # Changing the fingerprint algorithm invalidates every recorded cache and baseline.
    predicate = noul("is_urgent", question="Does `message` convey urgency?")
    assert (
        predicate.fingerprint
        == "sha256:"
        + __import__("hashlib")
        .sha256(b'{"criteria":null,"kind":"noul","question":"Does `message` convey urgency?"}')
        .hexdigest()
    )


def test_fingerprint_ignores_non_model_fields() -> None:
    base = noul("a", question="q?", criteria={"true": "t"})
    variants = [
        noul("renamed", question="q?", criteria={"true": "t"}),
        noul("a", question="q?", criteria={"true": "t"}, threshold=0.9),
        noul("a", question="q?", criteria={"true": "t"}, tags=["x"], description="doc", version="2"),
    ]
    assert {variant.fingerprint for variant in variants} == {base.fingerprint}


def test_fingerprint_tracks_model_facing_fields() -> None:
    base = noul("a", question="q?", criteria={"true": "t"})
    assert noul("a", question="q!", criteria={"true": "t"}).fingerprint != base.fingerprint
    assert noul("a", question="q?", criteria={"true": "t", "false": "f"}).fingerprint != base.fingerprint
    assert noul("a", question="q?").fingerprint != base.fingerprint


def test_choice_fingerprint_depends_on_option_order() -> None:
    first = choice("a", question="q?", options={"x": None, "y": None})
    second = choice("a", question="q?", options={"y": None, "x": None})
    assert first.fingerprint != second.fingerprint


def test_structured_question_key_order_does_not_matter() -> None:
    one = noul("a", question={"question": "q?", "focus": "f"})
    two = noul("a", question={"focus": "f", "question": "q?"})
    assert one.fingerprint == two.fingerprint


def test_kinds_never_collide() -> None:
    levels = ["low", "high"]
    choice_predicate = choice("a", question="q?", options={"low": None, "high": None})
    score_predicate = score("a", question="q?", levels=levels)
    assert choice_predicate.fingerprint != score_predicate.fingerprint


def test_set_fingerprint_covers_ids_but_not_order_or_thresholds() -> None:
    a = noul("a", question="q?")
    b = noul("b", question="r?")
    s1 = PredicateSet(name="s", predicates=(a, b))
    s2 = PredicateSet(name="s", predicates=(b, a))
    s3 = PredicateSet(name="s", predicates=(noul("a", question="q?", threshold=0.4), b))
    s4 = PredicateSet(name="s", predicates=(noul("renamed", question="q?"), b))
    assert s1.fingerprint == s2.fingerprint == s3.fingerprint
    assert s1.fingerprint != s4.fingerprint
    assert s1.predicate_fingerprints() == {"a": a.fingerprint, "b": b.fingerprint}


# ---- selection ----------------------------------------------------------------------------------


def test_select_keeps_set_order_and_rejects_unknown_ids() -> None:
    predicate_set = _set()
    assert predicate_set.select(["tone", "is_urgent"]).ids() == ("is_urgent", "tone")
    with pytest.raises(KeyError, match="no predicate 'nope'"):
        predicate_set.select(["nope"])


def test_of_wraps_sequences() -> None:
    predicate_set = PredicateSet.of([noul("a", question="q?")])
    assert predicate_set.name == "adhoc"
    assert PredicateSet.of(predicate_set) is predicate_set


# ---- files --------------------------------------------------------------------------------------


def test_yaml_round_trip_is_lossless(tmp_path: Path) -> None:
    predicate_set = _set()
    path = tmp_path / "demo.yaml"
    predicate_set.to_file(path)
    loaded = PredicateSet.from_file(path)
    assert loaded == predicate_set
    assert loaded.fingerprint == predicate_set.fingerprint
    assert "true:" not in path.read_text() or "'true'" not in path.read_text()


def test_reference_sets_round_trip(repo_root: Path) -> None:
    for path in sorted((repo_root / "predicates").glob("*.yaml")):
        predicate_set = PredicateSet.from_file(path)
        assert PredicateSet.from_yaml(predicate_set.to_yaml()) == predicate_set, path.name


def test_yaml_criteria_keys_are_written_naturally() -> None:
    predicate_set = PredicateSet(
        name="s", predicates=(noul("a", question="q?", criteria={"true": "t", "false": "f"}),)
    )
    text = predicate_set.to_yaml()
    assert "      true: t\n" in text
    assert "      false: f\n" in text


def test_yaml_option_keys_survive_yaml_scalars() -> None:
    text = """
format: sentinel/predicates/v1
name: s
predicates:
  answer:
    kind: choice
    question: Did they agree?
    options:
      yes: They agreed.
      no: They declined.
      true: Literally true.
      2: Two.
"""
    predicate = PredicateSet.from_yaml(text)["answer"]
    assert isinstance(predicate, ChoicePredicate)
    assert list(predicate.options) == ["yes", "no", "true", "2"]


def test_json_predicate_files_are_accepted(tmp_path: Path) -> None:
    path = tmp_path / "s.json"
    path.write_text(
        json.dumps(
            {
                "format": "sentinel/predicates/v1",
                "name": "s",
                "predicates": {"a": {"kind": "noul", "question": "q?", "criteria": {"true": "t"}}},
            }
        )
    )
    assert PredicateSet.from_file(path)["a"].question == "q?"


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("name: s\n", "expected `format: sentinel/predicates/v1`"),
        ("- a\n- b\n", "must be a mapping"),
        ("format: sentinel/predicates/v1\nname: s\npredicates: {}\n", "predicates"),
        (
            "format: sentinel/predicates/v1\nname: s\npredicates:\n  a: {kind: noul, question: q, threshld: 0.5}\n",
            "predicates.a.threshld: Extra inputs are not permitted",
        ),
        (
            "format: sentinel/predicates/v1\nname: s\npredicates:\n  a: {kind: choice, question: q, options: {x: null}}\n",
            "predicates.a.options",
        ),
        (
            "format: sentinel/predicates/v1\nname: s\npredicates:\n  a: {kind: bool, question: q}\n",
            "predicates.a",
        ),
        (
            "format: sentinel/predicates/v1\nname: s\npredicates:\n  a: {kind: noul, question: q}\n  a: {kind: noul}\n",
            "duplicate key",
        ),
        ("format: [unclosed\n", "invalid YAML"),
    ],
)
def test_file_errors_are_readable(text: str, fragment: str) -> None:
    with pytest.raises(PredicateValidationError) as caught:
        PredicateSet.from_yaml(text, source="f.yaml")
    assert fragment in str(caught.value)
    assert str(caught.value).startswith("f.yaml")


def test_missing_file_is_a_validation_error(tmp_path: Path) -> None:
    with pytest.raises(PredicateValidationError, match="cannot read"):
        PredicateSet.from_file(tmp_path / "absent.yaml")


def test_file_schema_exports() -> None:
    schema = PredicateFileV1.model_json_schema()
    assert schema["title"] == "Sentinel predicate file (v1)"
    assert "predicates" in schema["required"]
