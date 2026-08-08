from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion200d(Strategy):
    rationale = (
        "This strategy aims to capitalize on mean reversion in stock prices relative to a "
        "trailing 200-day simple moving average (SMA). By buying stocks that are significantly "
        "below their historical average and selling those above, it seeks to profit from the "
        "expected reversion of prices to more typical levels over time."
    )

    def __init__(self, window: int = 200, threshold: float = 0.05) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        sma = history.select(
            pl.col("symbol"), (pl.col("adj_close").sum() / pl.col("adj_close").count()).alias("sma")
        ).collect()

        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in sma["symbol"].to_list() or symbol not in closes.columns:
                continue

            last_close = view.latest_close()[symbol]
            latest_sma = sma.filter(pl.col("symbol") == symbol).select("sma").item()
            deviation = (last_close - latest_sma) / latest_sma

            if deviation < -self._threshold:
                signals[symbol] = 1.0
            elif deviation > self._threshold:
                signals[symbol] = -1.0

        weight = sum(abs(v) for v in signals.values()) ** (-1)
        return Signal(
            information_available_at=stamp, weights={s: w * weight for s, w in signals.items()}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest