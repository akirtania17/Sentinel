"""YAML loading with YAML 1.2 core-schema scalars.

PyYAML implements YAML 1.1, where ``yes``/``no``/``on``/``off`` are booleans, ``19:00`` is the
integer 1140, and ``2026-09-18`` becomes a ``date``. All three silently corrupt predicate files and
fixture states. This loader resolves only the YAML 1.2 core schema (``true``/``false``, ``null``,
decimal/octal/hex integers, floats) and leaves everything else a string. It also rejects duplicate
mapping keys, which PyYAML otherwise resolves by keeping the last one.
"""

import math
import re
from typing import Any

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode, Node, ScalarNode

_BOOL = re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$")
_NULL = re.compile(r"^(?:~|null|Null|NULL|)$")
_INT = re.compile(r"^(?:[-+]?[0-9]+|0o[0-7]+|0x[0-9a-fA-F]+)$")
_FLOAT = re.compile(
    r"^(?:[-+]?(?:\.[0-9]+|[0-9]+(?:\.[0-9]*)?)(?:[eE][-+]?[0-9]+)?|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$"
)


YAMLError = yaml.YAMLError
"""Base class of every error ``load_yaml`` raises for malformed input."""

_MISSING = object()


class _Loader(yaml.SafeLoader):
    """SafeLoader restricted to YAML 1.2 core-schema implicit scalars."""

    def construct_mapping(self, node: MappingNode, deep: bool = False) -> dict[Any, Any]:  # type: ignore[override]
        self.flatten_mapping(node)
        seen: dict[Any, Any] = {}
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=True)  # pyright: ignore[reportUnknownVariableType]
            try:
                previous = seen.get(key, _MISSING)
            except TypeError:
                continue
            if previous is not _MISSING:
                # In Python `True == 1`, so `true` and `1` would silently collapse into one key.
                problem = (
                    f"found duplicate key {key!r}"
                    if type(previous) is type(key)
                    else f"key {key!r} collides with key {previous!r}; quote one of them"
                )
                raise ConstructorError(
                    "while constructing a mapping", node.start_mark, problem, key_node.start_mark
                )
            seen[key] = key
        return super().construct_mapping(node, deep=deep)


def _construct_bool(loader: _Loader, node: Node) -> bool:
    assert isinstance(node, ScalarNode)
    return loader.construct_scalar(node).lower() == "true"


def _construct_int(loader: _Loader, node: Node) -> int:
    assert isinstance(node, ScalarNode)
    text = str(loader.construct_scalar(node))
    if text.startswith("0o"):
        return int(text[2:], 8)
    if text.startswith("0x"):
        return int(text[2:], 16)
    return int(text, 10)


def _construct_float(loader: _Loader, node: Node) -> float:
    assert isinstance(node, ScalarNode)
    text = str(loader.construct_scalar(node)).lower()
    if text.endswith(".inf"):
        return -math.inf if text.startswith("-") else math.inf
    if text == ".nan":
        return math.nan
    return float(text)


_Loader.yaml_implicit_resolvers = {}
_Loader.add_implicit_resolver("tag:yaml.org,2002:bool", _BOOL, list("tTfF"))
_Loader.add_implicit_resolver("tag:yaml.org,2002:null", _NULL, [*"~nN", ""])
_Loader.add_implicit_resolver("tag:yaml.org,2002:int", _INT, list("-+0123456789"))
_Loader.add_implicit_resolver("tag:yaml.org,2002:float", _FLOAT, list("-+0123456789."))
_Loader.add_constructor("tag:yaml.org,2002:bool", _construct_bool)
_Loader.add_constructor("tag:yaml.org,2002:int", _construct_int)
_Loader.add_constructor("tag:yaml.org,2002:float", _construct_float)


class _Dumper(yaml.SafeDumper):
    """SafeDumper that quotes any string either YAML 1.1 or the 1.2 loader would read as a non-string.

    It never emits anchors and aliases: files Sentinel writes are meant to be read and reviewed.
    """

    def ignore_aliases(self, data: Any) -> bool:
        return True


# The dumper quotes a string when its resolvers would read it back as another type. Adding the
# loader's YAML 1.2 resolvers on top of the default 1.1 ones makes output safe for both readers.
for _tag, _pattern, _first in (
    ("tag:yaml.org,2002:bool", _BOOL, list("tTfF")),
    ("tag:yaml.org,2002:null", _NULL, [*"~nN", ""]),
    ("tag:yaml.org,2002:int", _INT, list("-+0123456789")),
    ("tag:yaml.org,2002:float", _FLOAT, list("-+0123456789.")),
):
    _Dumper.add_implicit_resolver(_tag, _pattern, _first)


# Characters YAML 1.1 treats as line breaks or that plain and block styles cannot carry faithfully.
_NEEDS_ESCAPES = re.compile("[\x00-\x08\x0b-\x1f\x7f-\x9f\u2028\u2029\ufeff]")


def _represent_str(dumper: yaml.SafeDumper, data: str) -> ScalarNode:
    if _NEEDS_ESCAPES.search(data) or "\r" in data:
        style: str | None = '"'
    elif "\n" in data:
        style = "|"
    else:
        style = None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_Dumper.add_representer(str, _represent_str)


def load_yaml(text: str) -> Any:
    """Parse ``text`` as YAML (or JSON) with YAML 1.2 core-schema scalars."""
    return yaml.load(text, Loader=_Loader)  # noqa: S506 - _Loader derives from SafeLoader.


def dump_yaml(data: Any) -> str:
    """Serialize ``data`` to block-style YAML that ``load_yaml`` reads back unchanged."""
    return yaml.dump(
        data, Dumper=_Dumper, sort_keys=False, allow_unicode=True, width=100, default_flow_style=False
    )
