"""Conservative token estimation.

TypeSafe does not publish its tokenizer, so Sentinel estimates tokens from characters of the
serialized JSON it sends. The ratio is deliberately pessimistic: measured English JSON state runs at
roughly 3.9 characters per token, so dividing by 3.2 over-counts by about a fifth. Every provider
call records the estimate next to the actual count so the ratio can be tuned from real usage.
"""

import math

CHARS_PER_TOKEN = 3.2
"""Characters of serialized JSON assumed per token."""

PER_QUESTION_TOKENS = 20
"""Fixed tokens assumed for each question's envelope (type, keys, criteria scaffolding)."""


def estimate_tokens_for_chars(chars: int) -> int:
    """Estimate tokens for ``chars`` characters of serialized JSON."""
    if chars <= 0:
        return 0
    return math.ceil(chars / CHARS_PER_TOKEN)
