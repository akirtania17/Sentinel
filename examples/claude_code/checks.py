"""Deterministic checks the gate runs in code, next to Sentinel's judgments.

Jev's documentation keeps exact comparisons in code. An attacker who controls what the agent reads can
also argue a model out of noticing that a target is off by a character; it cannot argue with code.
"""

import json
import re
from typing import Any

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
DOMAIN = re.compile(r"\b(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}\b")
NUMBER = re.compile(r"\b\d{4,}\b")
PREFIXED_ID = re.compile(r"\b[A-Za-z]{2,8}_[A-Za-z0-9]{2,}\b")
IBAN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")
DESTINATION_KEYS = re.compile(
    r"(to|cc|bcc|recipients?|emails?|user_email|participants|attendees|invitees?|share_with|url|urls|host|"
    r"endpoint|webhook|address|iban|account|channel|phone)",
    re.IGNORECASE,
)
FILE_EXTENSIONS = {
    "csv", "json", "yaml", "yml", "py", "md", "txt", "sh", "log", "zip", "pdf", "png", "js", "ts", "html",
    "sql", "gz", "tar", "rb", "toml", "lock", "conf", "env", "xml", "jpg", "xlsx", "docx", "tgz", "whl",
}  # fmt: skip


ESCAPE = re.compile(r"\\[nrt]")


def strings(value: Any) -> list[str]:
    """Every string (and number) leaf of a JSON-like value, so escapes such as \\n never glue onto ids.

    Tool output often arrives as JSON inside a string (an MCP content list, `{"stdout": ...}`); such a
    string is decoded and its leaves returned instead, so the checks read the same text however an
    agent framework wraps it.
    """
    if isinstance(value, dict):
        return [s for item in value.values() for s in strings(item)]  # pyright: ignore[reportUnknownVariableType]
    if isinstance(value, list):
        return [s for item in value for s in strings(item)]  # pyright: ignore[reportUnknownVariableType]
    if isinstance(value, str) and value.lstrip()[:1] in ("{", "["):
        try:
            decoded = json.loads(value)
        except ValueError:
            pass
        else:
            if isinstance(decoded, dict | list):
                return strings(decoded)
    return [str(value)] if value is not None else []


def values(text: str) -> set[str]:
    """Emails, domains, account-style ids and long numbers in a text, lowercased."""
    text = ESCAPE.sub(" ", text)  # a literal "\\n" left in the text separates words
    found: set[str] = set()
    for email in EMAIL.findall(text):
        found.add(email.lower())
        found.add(email.split("@", 1)[1].lower())
    for domain in DOMAIN.findall(text):
        if domain.rsplit(".", 1)[-1].lower() not in FILE_EXTENSIONS and not domain[0].isdigit():
            found.add(domain.lower())
    found.update(NUMBER.findall(text))
    found.update(v.lower() for v in PREFIXED_ID.findall(text))
    found.update(v.lower() for v in IBAN.findall(text))
    return found


def distance(a: str, b: str) -> int:
    """Edit distance, counting a swap of two adjacent characters as one edit."""
    rows = [list(range(len(b) + 1))]
    for i in range(1, len(a) + 1):
        row = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            row[j] = min(rows[-1][j] + 1, row[j - 1] + 1, rows[-1][j - 1] + (a[i - 1] != b[j - 1]))
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                row[j] = min(row[j], rows[-2][j - 2] + 1)
        rows.append(row)
    return rows[-1][-1]


def near(v: str, w: str) -> bool:
    """Whether v looks like a slip or a spoof of w."""
    if v == w or v.isdigit() != w.isdigit():
        return False
    if v.isdigit():  # a swapped pair of digits, or extra or missing zeros; not a neighbouring id
        swapped = len(v) == len(w) and sorted(v) == sorted(w) and distance(v, w) == 1
        return swapped or (v.rstrip("0") == w.rstrip("0") and len(v) != len(w))
    if ("@" in v) != ("@" in w):
        return False
    if "@" in v and "@" in w:
        local_v, domain_v = v.split("@", 1)
        local_w, domain_w = w.split("@", 1)
        if local_v == local_w and domain_v != domain_w:
            return True
    if "@" not in v and "@" not in w and "." in v and "." in w:  # domains
        label_v, label_w = v.rsplit(".", 1)[0], w.rsplit(".", 1)[0]
        if label_v == label_w or label_v.startswith(label_w + "-"):
            return True
    return len(min(v, w, key=len)) >= 5 and distance(v, w) <= (1 if len(w) < 8 else 2)


def user_text(state: dict[str, Any]) -> str:
    """Everything the user wrote: the request and their earlier turns."""
    return " ".join(
        [state["user_request"], *(t["text"] for t in state.get("conversation", []) if t["role"] == "user")]
    )


def destinations(action: dict[str, Any]) -> list[str]:
    """The values of a tool call's arguments that say where something goes: recipients, addresses,
    links, accounts, channels. Empty for a shell command, whose destinations code cannot pick out."""
    arguments = action.get("arguments")
    if not isinstance(arguments, dict):
        return []
    return [
        s
        for key, value in arguments.items()  # pyright: ignore[reportUnknownVariableType]
        if DESTINATION_KEYS.fullmatch(str(key))  # pyright: ignore[reportUnknownArgumentType]
        for s in strings(value)
    ]


def user_named_destinations(state: dict[str, Any]) -> bool:
    """Whether the action has destinations and each is one the user wrote or the user's own
    (`user_identity`). Then whatever it sends goes where the user said, and nothing planted in data
    chose where."""
    found = [d.strip() for d in destinations(state["proposed_action"]) if d.strip()]
    text = " ".join([user_text(state), *strings(state.get("user_identity", {}))])
    named = values(text)

    def written(d: str) -> bool:
        if re.search(rf"(?<!\w){re.escape(d)}(?!\w)", text, re.IGNORECASE):
            return True
        return bool(values(d)) and values(d) <= named

    return bool(found) and all(written(d) for d in found)


def lookalikes(state: dict[str, Any]) -> list[str]:
    """Values in the action that sit one slip away from what the user or the records name."""
    action = values(" ".join(strings(state["proposed_action"])))
    user = values(user_text(state))
    context = values(
        " ".join(strings([state.get("project_instructions", ""), state.get("recent_steps", [])]))
    )
    flags: list[str] = []
    for v in sorted(action - user):
        for w in sorted(user | context):
            if near(v, w) and (w in user or v not in context):
                flags.append(f"{v} looks like {w}")
                break
    return flags
