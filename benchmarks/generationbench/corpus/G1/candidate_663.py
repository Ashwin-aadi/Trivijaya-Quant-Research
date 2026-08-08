from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Liquidity screening ensures that highly liquid stocks are given more weight in the "
        "portfolio. This strategy aims to balance risk and return by focusing on stocks with "
        "higher trading volumes."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volume_series = [float(v) for v in history["volume"].to_list()]
        symbol_list = [symbol for symbol in view.symbols if symbol in volume_series]
        liquidity_screened_closes = view.closes(lookback=self._window)

        if liquidity_screened_closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        closes_list = []
        for symbol in symbol_list:
            values = [float(v) for v in liquidity_screened_closes[symbol].drop_nulls().to_list()]
            if len(values) >= self._window:
                closes_list.append(symbol)

        weight = 1.0 / len(closes_list)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in closes_list}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest