from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion(Strategy):
    rationale = (
        "Price reversion occurs when prices return to a central tendency after deviating. "
        "By identifying symbols that have moved significantly from their trailing averages, "
        "we can generate buy or sell signals based on this reversion effect."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history["symbol"].unique()]
        signals: dict[str, float] = {}

        for symbol in symbols:
            df = (
                history.filter(pl.col("symbol") == symbol)
                .sort("session_date")
                .select(
                    pl.col("close").alias("price"),
                    (pl.col("close").shift(-1)).alias("previous_close"),
                )
            )

            if df.height < 2:
                continue

            close = float(df.select(pl.last("price")).item())
            previous_close = float(df.select(pl.last("previous_close")).item())

            price_change = (close - previous_close) / previous_close
            trailing_avg = (
                df.sort("session_date")
                .select(pl.col("price").mean().alias("trailing_avg"))
                .select(pl.first("trailing_avg"))
                .item()
            )

            reversion_score = abs(price_change - (close - trailing_avg)) / close

            if reversion_score > 0.05:  # Adjust the threshold as needed
                signals[symbol] = 1.0

        return Signal(
            information_available_at=stamp, weights={symbol: weight for symbol, weight in signals.items()}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest