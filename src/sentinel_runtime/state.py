"""Application state, canonicalized.

State is what predicates are judged against: a string, a JSON object, or a JSON array. Sentinel
never mutates the caller's object. ``State.of`` converts it into plain JSON containers (a deep
copy), applies the optional redaction hook to that copy, and records a digest and size estimate.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from sentinel_runtime.errors import StateError
from sentinel_runtime.hashing import canonical_json, sha256_digest
from sentinel_runtime.tokens import estimate_tokens_for_chars

JSONValue = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]
StateLike = str | Mapping[str, Any] | Sequence[Any] | BaseModel
"""What callers may pass as state: text, a mapping, a sequence, or a pydantic model."""

Redactor = Callable[[Any], Any]
"""Receives a plain-JSON copy of the state and returns the state to send."""


def _to_json(value: Any, path: str) -> JSONValue:
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise StateError("not_serializable", f"state{path} is {value}, which JSON cannot represent")
        return value
    if isinstance(value, BaseModel):
        return _to_json(value.model_dump(mode="json"), path)
    if isinstance(value, Mapping):
        out: dict[str, JSONValue] = {}
        for key, item in value.items():  # pyright: ignore[reportUnknownVariableType]
            if not isinstance(key, str):
                raise StateError(
                    "not_serializable",
                    f"state{path} has a non-string key {key!r}; JSON object keys must be strings",
                )
            out[key] = _to_json(item, f"{path}.{key}")
        return out
    if isinstance(value, list | tuple):
        return [_to_json(item, f"{path}[{index}]") for index, item in enumerate(value)]  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]
    raise StateError(
        "not_serializable",
        f"state{path} is a {type(value).__name__}; state must contain only JSON types "
        "(convert models with .model_dump(mode='json'), dates with .isoformat())",
    )


def _is_empty(value: JSONValue) -> bool:
    if isinstance(value, str):
        return not value.strip()
    return isinstance(value, dict | list) and not value


@dataclass(frozen=True, slots=True)
class State:
    """A canonicalized, immutable-by-convention state ready to send to a provider."""

    value: str | dict[str, Any] | list[Any]
    """Plain-JSON copy of the (redacted) state, in the caller's key order. This is what is sent."""
    canonical_json: str
    """Sorted-key canonical JSON of ``value``; the input to ``digest``."""
    digest: str
    """``sha256:<hex>`` of ``canonical_json``."""
    chars: int
    """Length of ``canonical_json`` in characters."""
    estimated_tokens: int
    """Conservative token estimate for ``value``."""

    @classmethod
    def of(
        cls, value: "StateLike | State", *, redact: Redactor | None = None, max_chars: int | None = None
    ) -> "State":
        """Canonicalize ``value``.

        Raises ``StateError`` when the state is empty, not JSON-representable, or longer than
        ``max_chars`` characters of canonical JSON.
        """
        if isinstance(value, State):
            if redact is None and (max_chars is None or value.chars <= max_chars):
                return value
            value = value.value
        plain = _to_json(value, "")
        if redact is not None:
            plain = _to_json(redact(plain), "")
        if not isinstance(plain, str | dict | list):
            raise StateError(
                "not_serializable",
                f"state must be a string, an object, or an array, not {type(plain).__name__}",
            )
        if _is_empty(plain):
            raise StateError("empty", "state is empty; there is nothing to judge")
        try:
            text = canonical_json(plain)
        except (TypeError, ValueError) as error:
            raise StateError("not_serializable", f"state is not JSON-serializable: {error}") from None
        try:
            digest = sha256_digest(text)
        except UnicodeEncodeError:
            raise StateError("not_serializable", "state contains text that is not valid Unicode") from None
        chars = len(text)
        if max_chars is not None and chars > max_chars:
            raise StateError(
                "too_large", f"state is {chars} characters of JSON, over the configured limit of {max_chars}"
            )
        return cls(
            value=plain,
            canonical_json=text,
            digest=digest,
            chars=chars,
            estimated_tokens=estimate_tokens_for_chars(chars),
        )
