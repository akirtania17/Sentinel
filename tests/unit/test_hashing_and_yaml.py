import math
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st
from yaml.constructor import ConstructorError

from sentinel_runtime._yaml import dump_yaml, load_yaml
from sentinel_runtime.hashing import canonical_json, digest_of, sha256_digest
from sentinel_runtime.tokens import estimate_tokens_for_chars

# Strings that YAML 1.1 or 1.2 would read back as another type if written unquoted.
yaml_lookalikes = st.sampled_from(
    [
        "0E0",
        "1e5",
        ".5",
        "+1",
        "0o7",
        "0x1F",
        "yes",
        "No",
        "on",
        "19:00",
        "2026-09-18",
        "~",
        "Null",
        ".inf",
        "\x85",
    ]
)
json_scalars = (
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text()
    | yaml_lookalikes
)
json_values = st.recursive(
    json_scalars,
    lambda children: (
        st.lists(children, max_size=4) | st.dictionaries(st.text(max_size=8), children, max_size=4)
    ),
    max_leaves=20,
)


@given(st.dictionaries(st.text(max_size=8), json_values, max_size=6))
def test_digest_ignores_key_order(value: dict[str, Any]) -> None:
    reversed_value = dict(reversed(list(value.items())))
    assert digest_of(value) == digest_of(reversed_value)


@given(json_values)
def test_canonical_json_is_deterministic(value: Any) -> None:
    assert canonical_json(value) == canonical_json(value)
    assert digest_of(value).startswith("sha256:")
    assert len(digest_of(value)) == len("sha256:") + 64


def test_canonical_json_rejects_nan_and_infinity() -> None:
    with pytest.raises(ValueError):
        canonical_json({"x": math.nan})
    with pytest.raises(ValueError):
        canonical_json([math.inf])


def test_canonical_json_is_compact_sorted_and_unicode() -> None:
    assert canonical_json({"b": 1, "a": ["é", None]}) == '{"a":["é",null],"b":1}'


def test_sha256_digest_known_value() -> None:
    assert sha256_digest("") == "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_token_estimate_is_monotonic_and_conservative() -> None:
    assert estimate_tokens_for_chars(0) == 0
    assert estimate_tokens_for_chars(1) == 1
    assert estimate_tokens_for_chars(3200) == 1000
    counts = [estimate_tokens_for_chars(n) for n in range(0, 500, 7)]
    assert counts == sorted(counts)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("yes", "yes"),
        ("no", "no"),
        ("on", "on"),
        ("off", "off"),
        ("19:00", "19:00"),
        ("2026-09-18", "2026-09-18"),
        ("true", True),
        ("False", False),
        ("null", None),
        ("~", None),
        ("", None),
        ("0755", 755),
        ("0o17", 15),
        ("0x1F", 31),
        ("-12", -12),
        ("1.5e3", 1500.0),
        ('"true"', "true"),
    ],
)
def test_yaml_uses_core_schema_scalars(text: str, expected: Any) -> None:
    assert load_yaml(f"value: {text}\n")["value"] == expected


def test_yaml_special_floats() -> None:
    data = load_yaml("a: .inf\nb: -.inf\nc: .nan\n")
    assert data["a"] == math.inf
    assert data["b"] == -math.inf
    assert math.isnan(data["c"])


def test_yaml_rejects_duplicate_keys() -> None:
    with pytest.raises(ConstructorError, match="duplicate key"):
        load_yaml("a: 1\nb: 2\na: 3\n")


def test_yaml_reads_json() -> None:
    assert load_yaml('{"a": [1, "two", null, true]}') == {"a": [1, "two", None, True]}


@given(st.dictionaries(st.text(min_size=1, max_size=10) | yaml_lookalikes, json_values, max_size=5))
def test_yaml_round_trip(value: dict[str, Any]) -> None:
    assert load_yaml(dump_yaml(value)) == value


def test_yaml_dumps_multiline_as_literal_block() -> None:
    assert "|" in dump_yaml({"text": "line one\nline two"})


def test_yaml_rejects_keys_that_collide_in_python() -> None:
    with pytest.raises(ConstructorError, match="collides with key True; quote one of them"):
        load_yaml("true: a\n1: b\n")
    assert load_yaml("'true': a\n1: b\n") == {"true": "a", 1: "b"}


def test_yaml_never_writes_aliases() -> None:
    shared = {"max": 1}
    text = dump_yaml({"a": shared, "b": shared})
    assert "&" not in text
    assert "*" not in text
    assert load_yaml(text) == {"a": {"max": 1}, "b": {"max": 1}}
