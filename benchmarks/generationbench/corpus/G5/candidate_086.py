from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "This strategy follows trends in assets with lower volatility. "
        "The idea is that during periods of low volatility, the market tends to move in a more linear fashion, "
        "and trends are likely to persist. We enter positions based on the asset's recent price action relative "
        "to its historical range."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window or len(history.columns) == 1:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history["symbol"].unique().to_list():
                continue
            adj_closes = history.filter(pl.col("symbol") == symbol)[
                "adj_close"
            ].drop_nulls()
            if adj_closes.height < self._window or adj_closes.is_empty():
                continue

            # Calculate the volatility-adjusted return
            returns = (adj_closes / adj_closes.shift(1) - 1.0).mean().to_list()[0]
            volatility = adj_closes.std().to_list()[0]

            if returns > 2 * volatility:
                picks.append(symbol)

        if not picks:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().to_list()[0]
    assert isinstance(newest, date)
    return newest