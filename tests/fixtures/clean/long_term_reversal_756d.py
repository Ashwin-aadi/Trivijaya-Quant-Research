"""Buy the names that have performed worst over roughly the past three years."""

from __future__ import annotations

from src.backtest.strategy import MarketView, Signal, Strategy

from ._common import equal_weight, latest_visible, top_n, window_return


class LongTermReversal756d(Strategy):
    """Long-horizon reversal: the opposite sign to the short-horizon momentum rules."""

    rationale = (
        "At horizons of three years and beyond, equity returns have historically reversed rather "
        "than continued, the usual explanation being that a long run of bad news drives valuation "
        "further than the fundamentals justify. This buys the worst three-year performers, which "
        "is deliberately the opposite sign to the shorter-horizon momentum rules in this set."
    )

    def __init__(self, lookback: int = 756, holdings: int = 10) -> None:
        self._lookback = lookback
        self._holdings = holdings

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        # Three years of history must genuinely exist. Without this guard window_return would
        # quietly measure whatever shorter span is available and the strategy would stop being a
        # long-horizon rule during the early part of any backtest.
        closes = view.closes(lookback=self._lookback + 1)
        if closes.height < self._lookback + 1:
            return Signal(information_available_at=stamp, weights={})

        returns = window_return(view, self._lookback)
        return Signal(
            information_available_at=stamp,
            weights=equal_weight(top_n(returns, self._holdings, largest=False)),
        )
