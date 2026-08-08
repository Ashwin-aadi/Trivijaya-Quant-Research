from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class TrailingReversion(Strategy):
    rationale = (
        "This strategy seeks to exploit mean reversion by identifying stocks that have "
        "fallen below their trailing 50-day moving average and are near the lower bound of "
        "their recent price range. Such stocks are likely to bounce back, offering buying "
        "opportunities."
    )

    def __init__(self, window: int = 50, buffer_days: int = 10) -> None:
        self._window = window
        self._buffer_days = buffer_days

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._buffer_days)

        if history.is_empty() or history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        signals: dict[str, float] = {}

        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue

            df = history.select(
                pl.col("session_date"),
                pl.col("symbol").alias("symbol"),
                pl.col("close").alias("price")
            ).filter(pl.col("symbol") == symbol)

            prices = [float(v) for v in df["price"].drop_nulls().to_list()]
            if len(prices) < self._window:
                continue

            moving_average = sum(prices[-self._window:]) / self._window
            latest_close = float(df.filter(pl.col("session_date") == stamp).select("close").item())

            # Calculate the range between recent high and low prices
            high_low_range = max(prices) - min(prices)

            if (
                latest_close < moving_average - (high_low_range * 0.25)
                and df.sort("session_date", descending=True).tail(self._buffer_days).height == self._buffer_days
            ):
                signals[symbol] = 1.0 / len(signals)

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