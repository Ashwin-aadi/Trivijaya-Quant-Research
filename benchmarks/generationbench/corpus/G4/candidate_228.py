from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MomentumStrategy(Strategy):
    rationale = (
        "This strategy exploits cross-sectional momentum by identifying stocks with strong past performance "
        "and allocating capital disproportionately towards these stocks. The assumption is that past winners are more likely to continue outperforming due to positive market sentiment and fundamental improvements."
    )

    def __init__(self, lookback_period: int = 120, top_n: int = 30) -> None:
        self._lookback_period = lookback_period
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_period)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        if len(symbols) < self._top_n:
            return Signal(information_available_at=stamp, weights={})

        prices = history.select(pl.col("session_date"), pl.col(symbols).arr.to_list())
        opens = history["open"][symbols].transpose().to_numpy()
        closes = history["close"][symbols].transpose().to_numpy()

        cumulative_returns = (closes / opens - 1.0) * 100
        vwap = _volume_weighted_average_price(history, symbols)

        scores = cumulative_returns.sum(axis=0)
        ranked_symbols = [symbol for symbol, score in sorted(zip(symbols, scores), key=lambda x: -x[1])[: self._top_n]]

        weight = 1.0 / len(ranked_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in ranked_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _volume_weighted_average_price(history: pl.DataFrame, symbols: list[str]) -> dict[str, float]:
    volumes = history["volume"][symbols].transpose().to_numpy()
    prices = history["close"][symbols].transpose().to_numpy()

    vwaps = (prices * volumes).sum(axis=0) / volumes.sum(axis=0)
    return {symbol: vwap for symbol, vwap in zip(symbols, vwaps)}