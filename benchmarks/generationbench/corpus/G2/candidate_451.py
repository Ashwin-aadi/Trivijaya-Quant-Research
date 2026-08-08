from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression suggests that when the trading range between the high and low "
        "shrinks significantly over a period, it often precedes an breakout or reversal. This "
        "strategy aims to identify such periods by measuring the ratio of daily range (high - low) "
        "to the 20-day average range."
    )

    def __init__(self, window: int = 20, threshold: float = 0.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)

        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history["symbol"].to_list()]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        ranges_df = (
            history.filter(pl.col("symbol").is_in(symbols))
                   .select(["symbol", "session_date", pl.col("high") - pl.col("low")])
                   .with_columns((pl.col("col_0") / pl.col("high").shift(-1) < self._threshold).alias("compress"))
        )

        compressed_symbols = []
        for symbol in symbols:
            if ranges_df.filter(pl.col("symbol") == symbol).select("compress").to_series().any():
                compressed_symbols.append(symbol)

        weight = 1.0 / len(compressed_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in compressed_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest