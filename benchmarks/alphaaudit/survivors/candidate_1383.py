from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Liquidity plays a crucial role in market efficiency. High liquidity can "
        "indicate strong interest and activity in the stock, which often leads to more stable "
        "prices and better execution of trades. This strategy focuses on equally weighting "
        "highly liquid stocks."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        liquidity_screened_symbols = history.filter(
            (pl.col("volume").rolling_sum(window_size=self._window).over("symbol")
             .sum() / pl.col("adj_close").rolling_max(window_size=self._window).over("symbol")
             > 1.0)
        )["symbol"].unique().to_list()

        if not liquidity_screened_symbols:
            return Signal(information_available_at=stamp, weights={})

        num_symbols = len(liquidity_screened_symbols)
        weight_per_symbol = 1.0 / num_symbols
        return Signal(
            information_available_at=stamp,
            weights={s: weight_per_symbol for s in liquidity_screened_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest