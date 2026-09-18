import datetime as dt
import math
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import BaseModel

from sentinel_runtime import State, StateError


def test_accepts_strings_objects_and_arrays() -> None:
    assert State.of("hello").value == "hello"
    assert State.of({"a": 1}).value == {"a": 1}
    assert State.of(["a", {"b": None}]).value == ["a", {"b": None}]


def test_tuples_become_lists_and_models_are_dumped() -> None:
    class Action(BaseModel):
        tool: str
        when: dt.date

    state = State.of({"steps": ("a", "b"), "action": Action(tool="send", when=dt.date(2026, 9, 18))})
    assert state.value == {"steps": ["a", "b"], "action": {"tool": "send", "when": "2026-09-18"}}


@pytest.mark.parametrize("empty", ["", "   \n", {}, []])
def test_rejects_empty_state(empty: Any) -> None:
    with pytest.raises(StateError) as caught:
        State.of(empty)
    assert caught.value.reason == "empty"


@pytest.mark.parametrize(
    ("value", "fragment"),
    [
        ({"when": dt.datetime(2026, 1, 1)}, "state.when is a datetime"),
        ({"n": math.nan}, "state.n is nan"),
        ({"x": [1, math.inf]}, "state.x[1] is inf"),
        ({1: "a"}, "non-string key 1"),
        ({"s": {1, 2}}, "state.s is a set"),
        (42, "must be a string, an object, or an array"),
        (None, "must be a string, an object, or an array"),
    ],
)
def test_rejects_non_json_state(value: Any, fragment: str) -> None:
    with pytest.raises(StateError) as caught:
        State.of(value)
    assert caught.value.reason == "not_serializable"
    assert fragment in str(caught.value)


def test_never_aliases_the_callers_object() -> None:
    original: dict[str, Any] = {"request": "hi", "items": [1, 2]}
    state = State.of(original)
    original["items"].append(3)
    original["request"] = "changed"
    assert state.value == {"request": "hi", "items": [1, 2]}
    assert state.digest == State.of({"request": "hi", "items": [1, 2]}).digest


def test_redactor_receives_a_copy_and_its_output_is_sent() -> None:
    original = {"email": "a@example.com", "body": "text"}

    def redact(value: Any) -> Any:
        value["email"] = "[redacted]"
        return value

    state = State.of(original, redact=redact)
    assert state.value == {"email": "[redacted]", "body": "text"}
    assert original["email"] == "a@example.com"


def test_redactor_output_is_validated() -> None:
    with pytest.raises(StateError, match="empty"):
        State.of({"a": 1}, redact=lambda _: {})


def test_max_chars() -> None:
    assert State.of("x" * 10, max_chars=12).chars == 12
    with pytest.raises(StateError) as caught:
        State.of("x" * 11, max_chars=12)
    assert caught.value.reason == "too_large"


def test_state_instances_pass_through() -> None:
    state = State.of({"a": 1})
    assert State.of(state) is state
    assert State.of(state, redact=lambda v: {**v, "b": 2}).value == {"a": 1, "b": 2}


def test_digest_ignores_key_order_but_value_keeps_it() -> None:
    first = State.of({"b": 1, "a": 2})
    second = State.of({"a": 2, "b": 1})
    assert first.digest == second.digest
    assert list(first.value) == ["b", "a"]
    assert first.canonical_json == '{"a":2,"b":1}'


def test_size_estimate() -> None:
    state = State.of({"text": "x" * 316})
    assert state.chars == len('{"text":""}') + 316
    assert state.estimated_tokens == math.ceil(state.chars / 3.2)


@given(
    st.dictionaries(
        st.text(min_size=1, max_size=6), st.integers() | st.text(max_size=10), min_size=1, max_size=6
    )
)
def test_digest_is_stable_for_equal_values(value: dict[str, Any]) -> None:
    assert State.of(value).digest == State.of(dict(sorted(value.items()))).digest
