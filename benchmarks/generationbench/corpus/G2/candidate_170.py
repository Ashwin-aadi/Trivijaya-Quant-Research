from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "High volatility often precedes price reversals. By identifying stocks with elevated "
        "volatility and maintaining a position for a trend, we can capture the upside during"
        " positive trends and avoid potential losses during negative ones."
    )

    def __init__(self, window: int = 20, volatility_threshold: float = 1.5) -> None:
        self._window = window
        self._volatility_threshold = volatility_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue

            # Calculate daily returns and rolling standard deviation
            returns = [(values[i] - values[i - 1]) / values[i - 1] for i in range(1, len(values))]
            std_dev = pl.Series(returns).rolling_std(window=self._window).to_list()[self._window - 1]

            # Check if the rolling standard deviation exceeds the threshold
            if std_dev > self._volatility_threshold:
                picks.append(symbol)

        picks = list(set(picks))[:20]  # Ensure no duplicates and limit to top 20
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest