from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionStrategy(Strategy):
    rationale = (
        "Price reversion occurs when prices move away from recent levels of support and "
        "resistance. This strategy exploits the tendency for prices to revert back towards "
        "these historical levels."
    )

    def __init__(self, window: int = 20, k: float = 1.5) -> None:
        self._window = window
        self._k = k

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        latest_closes = view.closes()
        symbols = [symbol for symbol in view.symbols if symbol in latest_closes.columns]

        # Calculate trailing average and standard deviation
        avg_close = history.group_by("symbol").agg(
            (pl.col("adj_close").mean()).alias("trailing_avg")
        )
        std_dev_close = history.group_by("symbol").agg(
            (pl.col("adj_close").std().alias("trailing_std"))
        )

        # Join to get the latest close and trailing stats
        combined = (
            avg_close.join(std_dev_close, on="symbol", how="inner")
            .join(latest_closes, on="symbol", how="inner")
        )
        combined = combined.with_columns(
            (pl.col("adj_close") - pl.col("trailing_avg")) / pl.col("trailing_std").alias("z_score")
        )

        # Filter out symbols that are far from their trailing mean
        filtered = combined.filter(pl.col("z_score").abs() > self._k)
        if filtered.height == 0:
            return Signal(information_available_at=stamp, weights={})

        # Calculate reversion weight for each symbol
        weights = {symbol: 1.0 / len(filtered) for symbol in symbols}
        return Signal(
            information_available_at=stamp, weights={s: weights[s] if s in filtered.columns else 0.0 for s in view.symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest