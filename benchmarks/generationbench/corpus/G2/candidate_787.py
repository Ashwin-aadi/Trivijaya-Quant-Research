from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression can indicate that volatility is about to increase. "
        "This suggests an entry point into sectors or stocks with compressed ranges before the "
        "potential for larger price movements."
    )

    def __init__(self, window: int = 60) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        range_compression = (
            history
            .group_by("symbol")
            .agg(
                (pl.col("high") - pl.col("low")).mean().alias("range_mean"),
                (pl.col("adj_close").shift(-1) - pl.col("adj_close")).abs().mean().alias("price_diff_mean"),
            )
        )

        symbols = range_compression.select(pl.col("symbol"))["symbol"]
        mean_ranges = range_compression.select(pl.col("range_mean"))["range_mean"].to_list()
        mean_prices = range_compression.select(pl.col("price_diff_mean"))["price_diff_mean"].to_list()

        def is_compressed(symbol, range_mean, price_diff_mean):
            symbol_history = history.filter(pl.col("symbol") == symbol)
            if symbol_history.height < self._window:
                return False
            latest_range = (symbol_history.select(pl.col("high").max()).item() - 
                            symbol_history.select(pl.col("low").min()).item())
            latest_price_diff = abs(symbol_history.select(pl.col("adj_close").shift(-1) - pl.col("adj_close")).mean().item())
            return latest_range / range_mean < 0.9 and latest_price_diff / price_diff_mean < 0.9

        compressed_symbols = [symbol for symbol, range_mean, price_diff_mean in zip(symbols, mean_ranges, mean_prices)
                              if is_compressed(symbol, range_mean, price_diff_mean)]

        if not compressed_symbols:
            return Signal(information_available_at=stamp, weights={})

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