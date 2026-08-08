from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class DispersionCompressionStrategy(Strategy):
    rationale = (
        "The strategy aims to capitalize on periods of high volatility (dispersion) and low price swings "
        "(range compression). By identifying stocks that exhibit these characteristics, we can capture intraday movements. "
        "High dispersion indicates potential for significant price movement, while range compression suggests a stable period."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty() or history.height < 2 * self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily range
        high_low_range = (
            history.select(
                pl.col("session_date"),
                (pl.col("high") - pl.col("low")).alias("range")
            )
            .with_columns(
                (pl.col("close").shift(-1) - pl.col("adj_close")).abs().alias("next_day_abs_change")
            )
            .sort("session_date", descending=False)
        )

        # Calculate historical average range
        avg_range = high_low_range.select(
            pl.col("range").mean().alias("avg_range")
        ).collect()["avg_range"][0]

        # Identify dispersion days
        dispersion_days = (
            high_low_range.filter(
                (pl.col("range") > 1.5 * avg_range) &
                (pl.col("next_day_abs_change") < 0.02)
            )
            .sort("session_date", descending=False)
            .select(pl.col("session_date"))
        )

        # Find top quintile of stocks with low ATR
        atr = (
            history.select(
                pl.col("symbol"),
                (pl.col("high") - pl.col("low")).mean().alias("atr")
            )
            .sort("atr", descending=False)
            .select(pl.col("symbol").head(self._top_n))
        )

        # Combine dispersion days and low ATR stocks
        picks: list[str] = [str for str in dispersion_days["session_date"].to_list() if str in atr.to_dict()["symbol"]]
        picks = picks[: self._top_n]
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