"""Equal-weight the constituents whose most recent visible close clears a minimum price.

Naming note: the local ``latest_close`` holds the dictionary returned by ``view.latest_close()``,
which is the last close *before* the decision date. "Latest" means latest visible, not latest in
the sample: the view has no access to any session on or after the date being traded.
"""

from __future__ import annotations

from src.backtest.strategy import MarketView, Signal, Strategy

from ...clean._common import equal_weight, latest_visible


class LatestClosePriceFloor(Strategy):
    """A price-level screen applied to the last close the strategy is allowed to see."""

    rationale = (
        "NSE quotes in five-paisa ticks, so a stock trading at twenty rupees moves in steps worth "
        "twenty-five basis points while one at two thousand moves in steps worth a quarter of "
        "one. A floor on price removes the names where tick granularity is a material share of "
        "the daily move. It is a data-hygiene filter and not a forecast: price level carries no "
        "information about future return, so this should perform like the universe it screens, "
        "minus whatever the excluded names contributed."
    )

    def __init__(self, min_price: float = 100.0) -> None:
        if min_price < 0.0:
            raise ValueError("the price floor cannot be negative")
        self._min_price = min_price

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        latest_close = view.latest_close()
        if not latest_close:
            return Signal(information_available_at=stamp, weights={})

        investable = set(view.symbols)
        picks = sorted(
            symbol
            for symbol, price in latest_close.items()
            if symbol in investable and price >= self._min_price
        )
        return Signal(information_available_at=stamp, weights=equal_weight(picks))
