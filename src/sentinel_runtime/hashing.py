"""Canonical JSON and content digests.

Every fingerprint and digest in Sentinel is ``sha256`` over the canonical JSON form of a value:
sorted object keys, compact separators, UTF-8, and no NaN or infinity. Two values that are equal
as JSON always produce the same digest, on every machine and Python version.
"""

import hashlib
import json
from typing import Any

DIGEST_PREFIX = "sha256:"


def canonical_json(value: Any) -> str:
    """Serialize ``value`` to canonical JSON.

    Raises ``TypeError`` for values JSON cannot represent and ``ValueError`` for NaN or infinity.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_digest(text: str) -> str:
    """Return ``"sha256:<hex>"`` for the UTF-8 encoding of ``text``."""
    return DIGEST_PREFIX + hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest_of(value: Any) -> str:
    """Return the digest of the canonical JSON form of ``value``."""
    return sha256_digest(canonical_json(value))
