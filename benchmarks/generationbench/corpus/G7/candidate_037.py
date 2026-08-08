from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion5d(Strategy):
    rationale = (
        "Short-horizon mean reversion aims to profit from the tendency of prices to revert "
        "to their average levels over a certain period. By identifying stocks that have moved"
        " significantly away from their 5-day moving average, we can exploit potential price reversals."
    )

    def __init__(self, window: int = 5) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.is_empty() or history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        symbols = [str(s) for s in view.symbols]
        closes = history.select(
            pl.col("symbol").cast(pl.Utf8),
            (pl.col("adj_close") - pl.col("adj_close").rolling_mean(window_size=self._window)).alias("deviation"),
        )

        picks: list[str] = []
        for symbol in symbols:
            if symbol not in closes.columns or symbol == "symbol":
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window + 1:
                continue
            current_deviation = values[-1]
            if abs(current_deviation) > max(abs(v) for v in values[:-1]):
                picks.append(symbol)

        picks = list(set(picks))[:5]
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest