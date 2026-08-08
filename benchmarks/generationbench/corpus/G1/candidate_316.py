from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class TrailingReversion(Strategy):
    rationale = (
        "Price reverts towards the mean after deviating from it. By identifying symbols that "
        "have deviated significantly in a short-term period and then revert back to their "
        "mean over a longer term, we can potentially capture profitable movements."
    )

    def __init__(self, window_short: int = 20, window_long: int = 100) -> None:
        self._window_short = window_short
        self._window_long = window_long

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window_long + self._window_short - 1)
        if history.height < self._window_long + self._window_short - 1:
            return Signal(information_available_at=stamp, weights={})

        symbols = view.symbols
        short_ma = (
            history.group_by("symbol")
            .agg((pl.col("adj_close").rolling_mean(self._window_short)).alias("short_ma"))
            .with_columns(pl.col("short_ma").shift(1).alias("prev_short_ma"))
            .select(["symbol", "session_date", "short_ma", "prev_short_ma"])
        )
        long_ma = (
            history.group_by("symbol")
            .agg((pl.col("adj_close").rolling_mean(self._window_long)).alias("long_ma"))
            .with_columns(pl.col("long_ma").shift(1).alias("prev_long_ma"))
            .select(["symbol", "session_date", "long_ma", "prev_long_ma"])
        )

        merged = short_ma.join(long_ma, on=["symbol", "session_date"], how="inner")
        candidates: list[str] = []
        for symbol in symbols:
            if symbol not in merged.columns or merged[symbol].is_empty():
                continue
            prev_short_ma = float(merged[merged["symbol"] == symbol]["prev_short_ma"].to_list()[0])
            prev_long_ma = float(merged[merged["symbol"] == symbol]["prev_long_ma"].to_list()[0])
            if (
                (prev_short_ma > prev_long_ma and merged[symbol]["short_ma"][0] < merged[symbol]["long_ma"][0]) or
                (prev_short_ma < prev_long_ma and merged[symbol]["short_ma"][0] > merged[symbol]["long_ma"][0])
            ):
                candidates.append(symbol)

        if not candidates:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(candidates)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in candidates},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest