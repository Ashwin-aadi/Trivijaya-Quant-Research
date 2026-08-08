from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend following aims to capture trending markets by scaling the position size "
        "based on the recent volatility. High volatility periods indicate stronger trends, which should be "
        "exploited with larger positions."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
            )
            .sort("session_date", descending=False)
            .drop_nulls()
        )

        # Compute volatility over the window
        volatility = history.select(
            (pl.col("return").std() * 100).alias("volatility")
        ).height

        if volatility == 0:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            returns = [float(v) for v in history[symbol].select("return").to_list()]
            if len(returns) < self._window:
                continue
            # Calculate the sum of absolute returns over the window
            trend_score = sum(abs(r) for r in returns)
            if trend_score > 0:
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