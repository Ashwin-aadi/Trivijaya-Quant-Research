from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Liquidity screening ensures that we focus on stocks with sufficient trading volume, "
        "which can help in executing trades without significantly impacting the stock price. Equal weighting across these stocks aims to diversify risk and capture market-wide performance."
    )

    def __init__(self, window: int = 20, liquidity_threshold: float = 1_000_000) -> None:
        self._window = window
        self._liquidity_threshold = liquidity_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        filtered_history = (
            history.select(["symbol", "volume"])
                   .filter(pl.col("volume") > self._liquidity_threshold)
        )
        if filtered_history.height < 1:
            return Signal(information_available_at=stamp, weights={})

        symbols = [row[0] for row in filtered_history.select("symbol").to_numpy()]
        weight = 1.0 / len(symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest