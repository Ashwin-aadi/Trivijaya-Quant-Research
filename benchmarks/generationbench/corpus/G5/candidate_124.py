from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionToTrailingAverage(Strategy):
    rationale = (
        "Price reversion to the mean is a well-known phenomenon. This strategy identifies "
        "overbought or oversold conditions and exploits them by going long on underperforming "
        "stocks and short on overperforming ones."
    )

    def __init__(self, window: int = 50) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty() or history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        # Calculate the trailing average
        avg_close = (
            history.group_by("symbol")
                   .agg((pl.col("adj_close").mean()).alias("trailing_avg"))
        )

        # Compare current close to the trailing average
        closes = view.closes(lookback=self._window)
        avg_closes = closes.join(avg_close, on="symbol", how="inner")

        signals: dict[str, float] = {}
        for symbol in symbols:
            latest_close = avg_closes[avg_closes["session_date"] == stamp][symbol].to_list()[0]
            trailing_avg = avg_closes[avg_closes["session_date"] == stamp]["trailing_avg"].to_list()[0]

            if latest_close > trailing_avg * 1.05:
                signals[symbol] = -0.2  # Short the symbol
            elif latest_close < trailing_avg * 0.95:
                signals[symbol] = 0.3   # Long the symbol

        return Signal(
            information_available_at=stamp, weights={s: w for s, w in signals.items() if w != 0}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest