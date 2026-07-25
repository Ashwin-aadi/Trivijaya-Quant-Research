"""Hold names that have risen on the greatest share of recent sessions."""

from __future__ import annotations

from src.backtest.strategy import MarketView, Signal, Strategy

from ._common import daily_returns, equal_weight, latest_visible, top_n


class TrendPersistence(Strategy):
    """Counts up-days rather than measuring total return."""

    rationale = (
        "Counting positive sessions describes how consistently a name has advanced, which is not "
        "the same as how far it advanced: one enormous day can dominate a total return without "
        "indicating any persistent trend. This ranks on consistency instead."
    )

    def __init__(self, lookback: int = 63, holdings: int = 10) -> None:
        self._lookback = lookback
        self._holdings = holdings

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        series = daily_returns(view, self._lookback)
        scores = {
            symbol: sum(1 for r in rets if r > 0) / len(rets)
            for symbol, rets in series.items()
            if rets
        }
        return Signal(
            information_available_at=stamp,
            weights=equal_weight(top_n(scores, self._holdings)),
        )
