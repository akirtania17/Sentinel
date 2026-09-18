"""Inventory bookkeeping for the refurbished furniture warehouse."""

from dataclasses import dataclass


@dataclass
class Item:
    sku: str
    name: str
    quantity: int
    unit_price_cents: int


def total_value_cents(items: list[Item]) -> int:
    """The value of everything in stock."""
    return sum(item.unit_price_cents for item in items)


def low_stock(items: list[Item], threshold: int = 3) -> list[str]:
    """SKUs at or below the reorder threshold."""
    return [item.sku for item in items if item.quantity < threshold]


def reserve(items: list[Item], sku: str, quantity: int) -> Item:
    """Take stock out for an order; refuse to go negative."""
    for item in items:
        if item.sku == sku:
            if quantity > item.quantity:
                raise ValueError(f"only {item.quantity} of {sku} left")
            item.quantity -= quantity
            return item
    raise KeyError(sku)
