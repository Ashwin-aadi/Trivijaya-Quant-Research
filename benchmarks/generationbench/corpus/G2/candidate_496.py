from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionToTrailingMean(Strategy):
    rationale = (
        "Price reversion occurs when a security's price returns to the mean of its recent "
        "prices. In this strategy, we identify stocks that have moved away from their trailing "
        "mean and are likely to revert back. This is based on the assumption that stock prices "
        "tend to move towards their historical average."
    )

    def __init__(self, window: int = 60, z_score_threshold: float = 2.0) -> None:
        self._window = window
        self._z_score_threshold = z_score_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window or not all(symbol in history.columns for symbol in view.symbols):
            return Signal(information_available_at=stamp, weights={})

        mean_close = history.groupby("symbol").agg(
            (pl.col("adj_close").mean().alias("trailing_mean"))
        )

        price_z_score = (
            history
            .join(mean_close, on="symbol", how="inner")
            .with_columns(
                ((pl.col("adj_close") - pl.col("trailing_mean")).abs() / pl.col("trailing_mean").std().alias("z_score"))
            )
            .sort("session_date", descending=True)
            .select(
                "symbol",
                "session_date",
                "z_score"
            )
        )

        if price_z_score.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        z_scores = [float(v) for v in price_z_score["z_score"].to_list()]

        if max(z_scores) > self._z_score_threshold or min(z_scores) < -self._z_score_threshold:
            symbols_to_trade = [
                symbol
                for symbol, score in zip(price_z_score["symbol"].to_list(), z_scores)
                if abs(score) > self._z_score_threshold
            ]
        else:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(symbols_to_trade)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in symbols_to_trade}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest