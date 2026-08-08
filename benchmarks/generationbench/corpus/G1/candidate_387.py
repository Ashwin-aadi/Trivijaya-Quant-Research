from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqual(Strategy):
    rationale = (
        "Liquidity screening ensures that the portfolio is composed of stocks with "
        "high trading volumes. Equal weighting provides a balanced approach to risk "
        "among selected stocks."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        liquidity_screened_symbols = _screen_by_liquidity(history, symbols)
        equal_weights = 1.0 / len(liquidity_screened_symbols)

        return Signal(
            information_available_at=stamp,
            weights={s: equal_weights for s in liquidity_screened_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _screen_by_liquidity(history: pl.DataFrame, symbols: list[str]) -> list[str]:
    liquidity_threshold = history.select(pl.col("volume").mean()).item()
    screened_symbols = [
        symbol
        for symbol in symbols
        if float(history.filter(pl.col("symbol") == symbol)["volume"].max()) >= liquidity_threshold
    ]
    return screened_symbols