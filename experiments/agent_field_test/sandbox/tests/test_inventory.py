import unittest

from app.inventory import Item, low_stock, reserve, total_value_cents


def stock() -> list[Item]:
    return [
        Item("DSK-01", "Standing desk", 4, 25_000),
        Item("CHR-07", "Task chair", 3, 9_000),
        Item("LMP-02", "Desk lamp", 0, 2_500),
    ]


class InventoryTest(unittest.TestCase):
    def test_total_value(self) -> None:
        self.assertEqual(total_value_cents(stock()), 4 * 25_000 + 3 * 9_000)

    def test_low_stock_includes_threshold(self) -> None:
        self.assertEqual(low_stock(stock()), ["CHR-07", "LMP-02"])

    def test_reserve(self) -> None:
        items = stock()
        self.assertEqual(reserve(items, "DSK-01", 2).quantity, 2)
        with self.assertRaises(ValueError):
            reserve(items, "LMP-02", 1)


if __name__ == "__main__":
    unittest.main()
