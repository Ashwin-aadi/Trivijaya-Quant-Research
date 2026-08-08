from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "High dispersion in stock prices across certain sectors or industries followed by "
        "range compression can be exploited through writing protective puts. This strategy aims "
        "to identify periods of high volatility using Average True Range (ATR) and target stocks "
        "with the highest ATR values for range compression."
    )

    def __init__(self, window: int = 20, top_n: int = 10) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate ATR for each symbol
        atr_values = {}
        for symbol in view.symbols:
            high_low_diff = (history[pl.col("high")] - pl.col("low")).abs()
            true_range = (
                pl.col("high") - pl.col("close").shift(1).fill_null(pl.lit(history["close"][0]))
            ).abs() + \
                         (pl.col("low") - pl.col("open")).abs() + \
                         high_low_diff

            atr = true_range.mean().item()
            atr_values[symbol] = atr

        # Rank symbols by ATR values
        ranked_symbols = sorted(atr_values.items(), key=lambda x: x[1], reverse=True)
        selected_symbols = [symbol for symbol, _ in ranked_symbols[:self._top_n]]

        if not selected_symbols:
            return Signal(information_available_at=stamp, weights={})

        # Assign equal weight to each selected stock
        weight_per_stock = 0.10 / len(selected_symbols)

        return Signal(
            information_available_at=stamp,
            weights={s: weight_per_stock for s in selected_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest