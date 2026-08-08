from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "This strategy combines two characteristics: the 20-day return and the relative strength "
        "index (RSI) over a 14-day period. A high return suggests strong momentum, while RSI above "
        "70 indicates potential oversold conditions."
    )

    def __init__(self, window_return: int = 20, window_rsi: int = 14) -> None:
        self._window_return = window_return
        self._window_rsi = window_rsi

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window_return + self._window_rsi)
        if history.height < self._window_return + self._window_rsi:
            return Signal(information_available_at=stamp, weights={})

        rsi_threshold = 70
        returns: dict[str, float] = {}
        for symbol in view.symbols:
            closes = history.select(pl.col("symbol").eq(symbol).alias("is_symbol")).select(
                pl.col("adj_close")
            ).to_series()
            if len(closes) < self._window_return + self._window_rsi:
                continue
            returns[symbol] = (closes[-1] / closes[0] - 1.0)

        rsi: dict[str, float] = {}
        for symbol in view.symbols:
            closes = history.select(pl.col("symbol").eq(symbol).alias("is_symbol")).select(
                pl.col("adj_close")
            ).to_series()
            if len(closes) < self._window_rsi:
                continue
            delta = closes.diff().drop_nulls().shift(-1)
            gain = delta.where(delta > 0.0, 0.0).sum() / (len(delta) - 1)
            loss = -delta.where(delta < 0.0, 0.0).sum() / (len(delta) - 1)
            rsi[symbol] = 100 - (100 / (1 + gain / loss))

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in returns or symbol not in rsi:
                continue
            if returns[symbol] > 0.1 and rsi[symbol] < rsi_threshold:
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
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest