from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion(Strategy):
    rationale = (
        "This strategy exploits mean reversion by identifying stocks whose current prices "
        "have deviated significantly from their 50-day and 200-day simple moving averages. "
        "Stocks that fall below the 50-day SMA by more than -2% or rise above the 200-day SMA by"
        "more than +2% are targeted for potential mean reversion trades."
    )

    def __init__(self, ma_short_window: int = 50, ma_long_window: int = 200, threshold: float = 0.02) -> None:
        self._ma_short_window = ma_short_window
        self._ma_long_window = ma_long_window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=max(self._ma_long_window, self._ma_short_window))
        if closes.height < max(self._ma_long_window, self._ma_short_window):
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in closes.columns]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        ma_short = (
            closes[symbols]
            .with_columns(
                (pl.col("adj_close").rolling_mean(self._ma_short_window)).alias(f"ma_{self._ma_short_window}")
            )
            .select([f"session_date", f"*", *[f"ma_{self._ma_short_window}"]])
        )

        ma_long = (
            closes[symbols]
            .with_columns(
                (pl.col("adj_close").rolling_mean(self._ma_long_window)).alias(f"ma_{self._ma_long_window}")
            )
            .select([f"session_date", f"*", *[f"ma_{self._ma_long_window}"]])
        )

        combined = (
            ma_short.join(ma_long, on="symbol")
            .with_columns(
                (pl.col("adj_close") - pl.col(f"ma_{self._ma_short_window}")).alias("short_deviation"),
                (pl.col("adj_close") - pl.col(f"ma_{self._ma_long_window}")).alias("long_deviation"),
            )
        )

        buy_signals = combined.filter(
            (pl.col("short_deviation") < -self._threshold * 100) & (pl.col("long_deviation") > self._threshold * 100)
        ).select(["symbol"])

        sell_signals = combined.filter(
            (pl.col("short_deviation") > self._threshold * 100) & (pl.col("long_deviation") < -self._threshold * 100)
        ).select(["symbol"])

        buy_symbols = [row["symbol"] for row in buy_signals.to_dicts()]
        sell_symbols = [row["symbol"] for row in sell_signals.to_dicts()]

        if not buy_symbols and not sell_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight_per_buy_symbol = 0.05 / len(buy_symbols)
        weight_per_sell_symbol = -0.05 / len(sell_symbols)

        weights = {symbol: weight_per_buy_symbol for symbol in buy_symbols}
        weights.update({symbol: weight_per_sell_symbol for symbol in sell_symbols})

        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest