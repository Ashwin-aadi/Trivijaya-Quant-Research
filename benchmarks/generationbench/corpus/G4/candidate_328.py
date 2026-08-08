from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion200d(Strategy):
    rationale = (
        "Stock prices tend to revert to their long-term average levels. This strategy "
        "identifies stocks that have deviated significantly from their 200-day moving "
        "average and exploits mean-reverting behavior by going long on cheap stocks and "
        "shorting expensive ones."
    )

    def __init__(self, window: int = 200, threshold: float = 1.1, top_n: int = 30) -> None:
        self._window = window
        self._threshold = threshold
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        sma_50 = history.select(
            pl.col("symbol"),
            (pl.col("adj_close").rolling_mean(window_size=50)).alias("sma_50")
        )
        sma_200 = history.select(
            pl.col("symbol"),
            (pl.col("adj_close").rolling_mean(window_size=self._window)).alias(f"sma_{self._window}")
        )

        merged = sma_50.join(sma_200, on="symbol")
        merged = merged.with_columns([
            ((pl.col("adj_close") - pl.col(f"sma_{self._window}")) / pl.col(f"sma_{self._window}") * 100).alias("deviation"),
            (((pl.col("adj_close").shift(1) - pl.col(f"sma_{self._window}").shift(1)) / pl.col(f"sma_{self._window}").shift(1)) * 100).alias("prev_deviation")
        ])

        latest_closes = view.closes().select(pl.columns(*view.symbols))
        merged = merged.join(latest_closes, on="symbol")

        if merged.is_empty():
            return Signal(information_available_at=stamp, weights={})

        merged = merged.with_columns(
            (pl.col("deviation") < -10).alias("cheap"),
            (pl.col("deviation") > 10).alias("expensive")
        )

        cheap_symbols = [s for s in view.symbols if merged.filter(pl.col(f"sma_{self._window}") == latest_closes[s].item()).get_column("cheap").sum() > 0]
        expensive_symbols = [s for s in view.symbols if merged.filter(pl.col(f"sma_{self._window}") == latest_closes[s].item()).get_column("expensive").sum() > 0]

        cheap_weights, expensive_weights = {}, {}
        for symbol in cheap_symbols:
            deviation = merged.select(
                pl.col("deviation").filter(pl.col("symbol") == symbol).first().item()
            )
            weight = 1.0 / len(cheap_symbols) * (self._threshold - deviation)
            cheap_weights[symbol] = max(weight, 0)

        for symbol in expensive_symbols:
            deviation = merged.select(
                pl.col("deviation").filter(pl.col("symbol") == symbol).first().item()
            )
            weight = 1.0 / len(expensive_symbols) * (deviation - self._threshold)
            expensive_weights[symbol] = max(weight, 0)

        weights = {**cheap_weights, **expensive_weights}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest