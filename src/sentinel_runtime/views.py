"""Views: the part of a state a predicate is allowed to see.

A predicate may declare ``sees``, a list of state paths such as ``user_request`` or
``recent_steps[].tool``. The runtime then sends that predicate only those parts of the state. This
is a trust boundary: text outside the view (tool output an attacker may have written, for example)
cannot reach the judgment, whatever it says. Predicates with the same view share one request, so a
set with two views costs two requests, sent concurrently.

Paths name mapping keys separated by dots. ``[]`` after a name marks a list whose items are each
narrowed by the rest of the path; lists are narrowed item by item whether or not it is written, but
writing it keeps predicate files readable. A path that is absent from a state selects nothing.
"""

import copy
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Any, cast

from pydantic import StringConstraints

from sentinel_runtime.errors import StateError

STATE_PATH_PATTERN = r"^[A-Za-z_][\w-]*(\[\])?(\.[A-Za-z_][\w-]*(\[\])?)*$"
StatePath = Annotated[str, StringConstraints(pattern=STATE_PATH_PATTERN)]
"""A path into a state: dot-separated keys, each optionally followed by ``[]`` for a list."""

_Tree = dict[str, "_Tree | None"]
"""Keys to keep; ``None`` keeps the whole value under a key."""


def segments(path: str) -> list[str]:
    """The keys along a path, without list markers."""
    return [part.removesuffix("[]") for part in path.split(".")]


def _tree(paths: Iterable[str]) -> _Tree:
    root: _Tree = {}
    for path in paths:
        node = root
        keys = segments(path)
        for depth, key in enumerate(keys):
            last = depth == len(keys) - 1
            if key in node and node[key] is None:
                break  # an ancestor is already kept whole
            if last:
                node[key] = None
                break
            child = node.get(key)
            if child is None:
                child = {}
                node[key] = child
            node = child
    return root


_MISSING = object()


def _narrow(value: Any, tree: _Tree | None) -> Any:
    if tree is None:
        return copy.deepcopy(value)
    if isinstance(value, list):
        items = (_narrow(item, tree) for item in cast("list[Any]", value))
        return [item for item in items if item is not _MISSING]
    if not isinstance(value, Mapping):
        return _MISSING
    mapping = cast("Mapping[str, Any]", value)
    out: dict[str, Any] = {}
    for key, subtree in tree.items():
        if key in mapping:
            narrowed = _narrow(mapping[key], subtree)
            if narrowed is not _MISSING:
                out[key] = narrowed
    return out


def project(state: Any, paths: Sequence[str]) -> dict[str, Any]:
    """The parts of ``state`` that ``paths`` select, in a new object; the input is not modified.

    Raises ``StateError`` when the state is not a JSON object, since a view needs named fields.
    """
    if not isinstance(state, Mapping):
        raise StateError(
            "not_an_object", "a predicate that declares `sees` needs a state that is a JSON object"
        )
    narrowed = _narrow(state, _tree(paths))
    return cast("dict[str, Any]", narrowed)


def covers(paths: Sequence[str], reference: str) -> bool:
    """Whether a view over ``paths`` shows any of ``reference`` (a path a question names, which may
    use indexes such as ``recent_steps[0].tool``)."""
    keys = [key for key in re.split(r"\.|\[\d*\]", reference) if key]
    for path in paths:
        seen = segments(path)
        shared = min(len(seen), len(keys))
        if seen[:shared] == keys[:shared]:
            return True
    return False


__all__ = ["STATE_PATH_PATTERN", "StatePath", "covers", "project", "segments"]
