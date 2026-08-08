from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "Trend following strategies aim to capture the momentum in the market by buying "
        "assets that are trending upwards and selling those that are trending downwards. "
        "Volatility scaling adjusts position sizes based on recent volatility, allowing for "
        "larger positions during periods of low volatility when the market is more predictable."
    )

    def __init__(self, window: int = 20, z_score_threshold: float = 1.5) -> None:
        self._window = window
        self._z_score_threshold = z_score_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = history.with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
        )

        # Compute rolling mean and standard deviation of returns for volatility scaling
        history = history.with_columns(
            (
                pl.col("return").rolling_mean(window=self._window, by="symbol")
            ).alias("mean_return"),
            (pl.col("return").rolling_std(window=self._window, by="symbol")).alias(
                "volatility"
            ),
        )

        # Calculate z-score for each symbol
        history = history.with_columns(
            (
                (pl.col("return") - pl.col("mean_return")) / pl.col("volatility")
            ).alias("z_score")
        )

        # Identify trending symbols based on z-scores
        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            recent_history = history.select(
                [pl.col("symbol"), pl.col("z_score")]
            ).filter(pl.col("symbol") == symbol).sort("session_date", descending=True)
            if len(recent_history) < self._window:
                continue

            z_scores = [float(v) for v in recent_history["z_score"].to_list()]
            if all(z >= -self._z_score_threshold for z in z_scores):
                picks.append(symbol)

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest