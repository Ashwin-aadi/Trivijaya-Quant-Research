from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion20d(Strategy):
    rationale = (
        "Short-horizon mean reversion exploits the tendency of asset prices to return to their "
        "historical average. If an asset's price deviates significantly from its historical mean, "
        "it is likely to revert back to that level."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        mean_close = history.group_by("symbol").agg(
            (pl.col("adj_close").mean()).alias("mean_adj_close")
        )
        symbol_history = history.select(["session_date", "symbol", "adj_close"])
        merged = symbol_history.join(mean_close, on="symbol", how="inner")

        deviations = (
            (merged["adj_close"] - merged["mean_adj_close"]).abs() / merged["mean_adj_close"]
        ).alias("deviation")
        merged = merged.with_columns(deviations)

        for symbol in view.symbols:
            recent_closes = history.filter(pl.col("symbol") == symbol).select(
                "session_date", "adj_close"
            )
            if not recent_closes.height:
                continue
            latest_deviation = float(merged.filter(pl.col("symbol") == symbol)[deviations].last())
            if latest_deviation < 1 / self._threshold:
                return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(view.symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in view.symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest