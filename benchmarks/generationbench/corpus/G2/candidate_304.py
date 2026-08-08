from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionToTrailingMean(Strategy):
    rationale = (
        "Prices that revert to their trailing mean suggest that the current price is an "
        "overreaction. By buying below this mean and selling above it, one can profit from "
        "mean reversion tendencies in the market."
    )

    def __init__(self, window: int = 50) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        mean_price = (
            history.group_by("symbol")
                   .agg(pl.col("adj_close").mean().alias("mean_price"))
        )
        latest_prices = view.closes(lookback=self._window)

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in mean_price["symbol"].to_list():
                continue

            mean_value = float(mean_price.filter(
                pl.col("symbol") == symbol
            )["mean_price"][0])

            latest_close = float(latest_prices[symbol].drop_nulls().to_list()[-1])
            relative_distance = (latest_close - mean_value) / mean_value

            if relative_distance > 0.2:
                signals[symbol] = -relative_distance * 0.5
            elif relative_distance < -0.2:
                signals[symbol] = relative_distance * 0.5

        total_weight = sum(signals.values())
        if total_weight == 0:
            return Signal(information_available_at=stamp, weights={})

        adjusted_signals = {s: w / total_weight for s, w in signals.items()}
        return Signal(
            information_available_at=stamp,
            weights=adjusted_signals
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest