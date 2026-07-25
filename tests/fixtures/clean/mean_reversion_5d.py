"""Buy the past week's worst performers, expecting a bounce."""

from __future__ import annotations

from src.backtest.strategy import MarketView, Signal, Strategy

from ._common import equal_weight, latest_visible, top_n, window_return


class MeanReversion5d(Strategy):
    """Short-horizon reversal on one week of returns."""

    rationale = (
        "Over horizons of a few days, sharp moves are often driven by liquidity demand rather "
        "than news, and tend to partially reverse once that demand is satisfied. The portfolio "
        "buys the largest recent decliners."
    )

    def __init__(self, lookback: int = 5, holdings: int = 10) -> None:
        self._lookback = lookback
        self._holdings = holdings

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        scores = window_return(view, self._lookback)
        # largest=False: the worst recent returns are the buys.
        return Signal(
            information_available_at=stamp,
            weights=equal_weight(top_n(scores, self._holdings, largest=False)),
        )
