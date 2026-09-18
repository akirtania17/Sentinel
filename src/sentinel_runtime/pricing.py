"""Model prices and cost estimates.

Prices change, so Sentinel never stores cost as truth: evaluations record token counts, and cost is
computed from a dated price table when reported. Override or extend the table in ``sentinel.toml``::

    [pricing."jev-1.13.0"]
    input_per_mtok = 0.042
    output_per_mtok = 0.0
    as_of = "2026-09-17"
"""

from collections.abc import Mapping

from sentinel_runtime.config import PriceEntry, SentinelConfig

DEFAULT_PRICES: Mapping[str, PriceEntry] = {
    "jev-1.13.0": PriceEntry(input_per_mtok=0.042, output_per_mtok=0.0, as_of="2026-09-17"),
}
"""Published prices at the time of this release, keyed by versioned model id."""


class PricingTable:
    """Look up prices by the versioned model id a provider reports."""

    def __init__(
        self, prices: Mapping[str, PriceEntry] | None = None, *, include_defaults: bool = True
    ) -> None:
        self._prices: dict[str, PriceEntry] = dict(DEFAULT_PRICES) if include_defaults else {}
        self._prices.update(prices or {})

    @classmethod
    def from_config(cls, config: SentinelConfig) -> "PricingTable":
        """The default table with ``sentinel.toml`` overrides applied."""
        return cls(config.pricing)

    def price(self, model: str) -> PriceEntry | None:
        """The price entry for ``model``, or ``None`` when unknown."""
        return self._prices.get(model)

    def cost(self, model: str, input_tokens: int | None, output_tokens: int | None) -> float | None:
        """Estimated cost in US dollars, or ``None`` when the model or the input token count is unknown."""
        entry = self.price(model)
        if entry is None or input_tokens is None:
            return None
        total = input_tokens * entry.input_per_mtok + (output_tokens or 0) * entry.output_per_mtok
        return total / 1_000_000

    def models(self) -> tuple[str, ...]:
        """Models with a known price."""
        return tuple(self._prices)
