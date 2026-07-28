"""Equal-weight every point-in-time constituent that has a full window of visible prices.

Naming note: the local ``all_symbols`` is the complete universe on the decision date, read from
``view.symbols``. "All" means all constituents as at that date — a list that still contains names
which were later delisted or dropped from the index — and not the set of names that happened to
survive to the end of the sample.
"""

from __future__ import annotations

from src.backtest.strategy import MarketView, Signal, Strategy

from ...clean._common import equal_weight, latest_visible


class AllSymbolsEqualWeight(Strategy):
    """A passive baseline: hold the whole point-in-time universe, evenly."""

    rationale = (
        "This has no view on any individual name. It exists as the reference every other "
        "strategy is measured against: an equal-weighted holding of whatever the index contained "
        "on the day, rebalanced whenever the engine asks for a signal. Names without enough "
        "visible price history are skipped, because a recently listed name has no basis for the "
        "cost and liquidity assumptions the engine makes about it."
    )

    def __init__(self, min_history: int = 21) -> None:
        self._min_history = min_history

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        all_symbols = sorted(view.symbols)
        if not all_symbols:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._min_history)
        eligible = [
            symbol
            for symbol in all_symbols
            if symbol in closes.columns
            and closes[symbol].drop_nulls().len() >= self._min_history
        ]
        return Signal(information_available_at=stamp, weights=equal_weight(eligible))
