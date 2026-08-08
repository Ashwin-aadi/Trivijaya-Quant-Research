from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Short-horizon mean reversion assumes that prices which are far from their recent "
        "average will revert to it. This strategy buys undervalued stocks and sells overvalued ones."
    )

    def __init__(self, window: int = 10, deviation_threshold: float = 1.0) -> None:
        self._window = window
        self._deviation_threshold = deviation_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        avg_close = (
            history.select(pl.col("adj_close").mean().alias("avg"))
            .with_column(pl.lit(stamp).alias("session_date"))
            .select(["symbol", "session_date", "avg"])
        )

        latest_closes = view.closes(lookback=self._window)
        avg_close = avg_close.join(latest_closes, on="symbol")

        deviations = (avg_close["adj_close"] - avg_close["avg"]).abs().alias("deviation")
        avg_close = avg_close.select(["symbol", "deviation"])

        sorted_avg_close = (
            avg_close.sort("deviation", descending=False)
            .group_by("symbol")
            .agg(pl.col("deviation").mean().alias("mean_deviation"))
        )

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in sorted_avg_close.columns:
                continue
            deviation = float(sorted_avg_close[sorted_avg_close["symbol"] == symbol]["mean_deviation"])
            if deviation >= self._deviation_threshold:
                picks.append(symbol)

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