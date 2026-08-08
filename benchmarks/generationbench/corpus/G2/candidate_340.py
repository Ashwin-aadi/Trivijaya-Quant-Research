from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class TrailingReversion(Strategy):
    rationale = (
        "Price reversion occurs when a security's price deviates from its long-term mean. "
        "By identifying stocks that have moved significantly away from their historical "
        "mean, we can anticipate a return to the mean as market sentiment adjusts."
    )

    def __init__(self, window: int = 100) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbol_mean_close = (
            history.group_by("symbol")
            .agg(pl.col("adj_close").mean().alias("mean_adj_close"))
            .with_columns(
                (pl.col("adj_close") / pl.col("mean_adj_close") - 1.0).alias("z_score")
            )
        )

        recent_closes = view.closes(lookback=self._window)
        z_scores = symbol_mean_close.join(recent_closes, on="symbol").select(
            "symbol", "z_score"
        ).to_dicts()

        symbols_to_trade: list[str] = []
        for item in z_scores:
            if abs(item["z_score"]) > 1.0:
                symbols_to_trade.append(item["symbol"])

        weight_per_symbol = 1.0 / len(symbols_to_trade)
        weights = {s: weight_per_symbol for s in symbols_to_trade}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest