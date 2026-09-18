"""Deterministic checks the gate runs in code, next to Sentinel's judgments.

Jev's documentation keeps exact comparisons in code. An attacker who controls what the agent reads can
also argue a model out of noticing that a target is off by a character; it cannot argue with code.
"""

import re
from typing import Any

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
DOMAIN = re.compile(r"\b(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}\b")
NUMBER = re.compile(r"\b\d{4,}\b")
PREFIXED_ID = re.compile(r"\b[A-Za-z]{2,8}_[A-Za-z0-9]{2,}\b")
FILE_EXTENSIONS = {
    "csv", "json", "yaml", "yml", "py", "md", "txt", "sh", "log", "zip", "pdf", "png", "js", "ts", "html",
    "sql", "gz", "tar", "rb", "toml", "lock", "conf", "env", "xml", "jpg", "xlsx", "docx", "tgz", "whl",
}  # fmt: skip


def strings(value: Any) -> list[str]:
    """Every string (and number) leaf of a JSON-like value, so escapes such as \\n never glue onto ids."""
    if isinstance(value, dict):
        return [s for item in value.values() for s in strings(item)]  # pyright: ignore[reportUnknownVariableType]
    if isinstance(value, list):
        return [s for item in value for s in strings(item)]  # pyright: ignore[reportUnknownVariableType]
    return [str(value)] if value is not None else []


def values(text: str) -> set[str]:
    """Emails, domains, account-style ids and long numbers in a text, lowercased."""
    found: set[str] = set()
    for email in EMAIL.findall(text):
        found.add(email.lower())
        found.add(email.split("@", 1)[1].lower())
    for domain in DOMAIN.findall(text):
        if domain.rsplit(".", 1)[-1].lower() not in FILE_EXTENSIONS and not domain[0].isdigit():
            found.add(domain.lower())
    found.update(NUMBER.findall(text))
    found.update(v.lower() for v in PREFIXED_ID.findall(text))
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


def lookalikes(state: dict[str, Any]) -> list[str]:
    """Values in the action that sit one slip away from what the user or the records name."""
    action = values(" ".join(strings(state["proposed_action"])))
    user_text = " ".join(
        [state["user_request"], *(t["text"] for t in state.get("conversation", []) if t["role"] == "user")]
    )
    user = values(user_text)
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
