from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "This strategy identifies trending stocks by analyzing the ratio of recent "
        "price changes to their volatility. Higher ratios indicate stronger trends, "
        "which we exploit for potential gains."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            prices = [float(v) for v in history[symbol].drop_nulls().to_list()]
            price_changes = [(prices[i+1] - prices[i]) / max(prices[i], 1e-6) for i in range(len(prices) - 1)]
            volatility = pl.Series(price_changes).std()
            if volatility == 0:
                continue
            trend_ratio = price_changes[-1] / volatility

            if trend_ratio > self._threshold:
                signals[symbol] = 1.0 / len(signals)

        return Signal(
            information_available_at=stamp, weights={s: w for s, w in signals.items() if w}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest