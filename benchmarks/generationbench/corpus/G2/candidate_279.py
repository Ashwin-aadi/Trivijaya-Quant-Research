from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Liquidity is a proxy for marketability and information efficiency. "
        "Highly liquid stocks are more likely to be correctly priced, so they may provide returns that "
        "outperform the market when equally weighted."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_screened = (
            history.groupby("symbol")
                   .agg(pl.col("volume").mean().alias("avg_volume"))
                   .sort("avg_volume", descending=True)
                   .head(10)  # Select top 10 by average volume
        )

        symbols_to_trade = [row["symbol"] for row in liquidity_screened.to_dicts()]
        if not symbols_to_trade:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(symbols_to_trade)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in symbols_to_trade},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest