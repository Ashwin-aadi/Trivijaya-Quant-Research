from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression indicates a period of reduced volatility. During such periods, "
        "prices tend to stay within a narrow range, suggesting that investors are less active, "
        "potentially leading to mean reversion opportunities."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s in view.symbols if s in history.columns]
        if len(symbols) < 2:
            return Signal(information_available_at=stamp, weights={})

        means = history.select(
            pl.col("symbol").alias("symbol"),
            (pl.col("high") + pl.col("low")) / 2.0.alias("avg_price"),
        ).pivot(values="avg_price", index="symbol", aggregate_fn=None)

        ranges = []
        for symbol in symbols:
            high_min, low_max = (
                float(history.select(pl.col(symbol).max()).item()),
                float(history.select(pl.col(symbol).min()).item()),
            )
            range_compression = (high_min - low_max) / history.shape[0]
            ranges.append((symbol, range_compression))

        sorted_ranges = sorted(ranges, key=lambda x: x[1], reverse=True)
        top_symbols = [s for s, _ in sorted_ranges[:3]]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest