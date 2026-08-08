from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion200d(Strategy):
    rationale = (
        "This strategy aims to capitalize on mean-reverting behavior in stock prices "
        "relative to a trailing 200-day moving average. Stocks that have deviated significantly"
        " from their long-term equilibrium are expected to revert, providing profitable trading opportunities."
    )

    def __init__(self, window: int = 200, top_n: int = 50) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = history["adj_close"]
        sma_200 = (closes.rank(method="dense") - 1).rolling_sum(window=self._window) / self._window
        sma_200 = sma_200.fill_null(strategy="forward").alias("sma_200")

        history_with_sma = history.with_columns(sma_200)
        deviations = (history["adj_close"] - history_with_sma["sma_200"]).abs().alias("deviation")

        ranked_deviations = (
            history_with_sma
            .select(["symbol", "session_date", "deviation"])
            .sort("deviation", descending=True)
            .head(self._top_n)
        )

        picks: list[str] = [row["symbol"] for row in ranked_deviations.to_dict(as_series=False).values()]

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest