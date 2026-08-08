from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion20d(Strategy):
    rationale = (
        "Mean reversion is a market anomaly suggesting that asset prices and financial returns "
        "tend to move towards an average or mean over time. In short horizons, recent extreme "
        "highs or lows in stock prices are likely to revert back toward their historical mean."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        mean_close = (
            history.groupby("symbol")
                   .agg(pl.col("adj_close").mean().alias("mean_close"))
                   .select(["symbol", "mean_close"])
                   .with_columns((pl.col("adj_close") - pl.col("mean_close")).alias("deviation"))
        )

        recent_closes = view.closes(lookback=self._window)
        if recent_closes.is_empty():
            return Signal(information_available_at=stamp, weights={})

        signals: dict[str, float] = {}
        for symbol in mean_close["symbol"]:
            if symbol not in recent_closes.columns:
                continue
            deviation = float(recent_closes[symbol][-1]) - float(mean_close.filter(pl.col("symbol") == symbol)["mean_close"])
            if abs(deviation) > self._threshold * history.filter(pl.col("symbol") == symbol)["adj_close"].std():
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