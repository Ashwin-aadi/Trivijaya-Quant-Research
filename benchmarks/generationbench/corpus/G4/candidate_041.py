from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "This strategy identifies stocks that have recently broken above their 50-day moving average "
        "with strong volume. Once a breakout is confirmed, it sets a target price and places an order "
        "to buy or sell at the breakout level with defined stop-loss and take-profit levels."
    )

    def __init__(self, window: int = 5, top_n: int = 20) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate 50-day moving average
        ma_50 = history.groupby("symbol").agg(
            (pl.col("adj_close").sum() / pl.col("volume")).alias("ma_50")
        ).with_columns((pl.col("adj_close") - pl.col("ma_50")).alias("diff"))

        # Filter symbols with a strong breakout
        breakouts = ma_50.with_columns(
            (pl.col("diff").abs() / pl.col("volume")).alias("breakout_strength")
        ).sort("breakout_strength", descending=True).select(["symbol", "diff"])

        if breakouts.height < self._top_n:
            return Signal(information_available_at=stamp, weights={})

        # Select top N symbols with strong breakouts
        top_breakouts = [row["symbol"] for row in breakouts.to_dicts()[:self._top_n]]

        # Calculate target price and confirm breakout with volume
        closes = view.closes(lookback=self._window)
        signals: list[tuple[str, float]] = []

        for symbol in top_breakouts:
            if symbol not in closes.columns:
                continue

            close_values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            ma_50_value = history.filter(pl.col("symbol") == symbol).select("ma_50").to_numpy()[0][0]

            if len(close_values) < self._window:
                continue

            last_close = close_values[-1]
            second_last_close = close_values[-2]
            volume_ratio = close_values[-1] / close_values[-2]

            if last_close > ma_50_value and last_close > second_last_close and volume_ratio > 1.2:
                # Set target price
                range_high = max(close_values)
                range_low = min(close_values)
                typical_range = (range_high - range_low) / 2
                target_price = last_close + typical_range

                signals.append((symbol, target_price))

        if not signals:
            return Signal(information_available_at=stamp, weights={})

        weight_per_stock = 1.0 / len(signals)
        signal_weights = {symbol: weight for symbol, _ in signals}

        return Signal(
            information_available_at=stamp,
            weights=signal_weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest