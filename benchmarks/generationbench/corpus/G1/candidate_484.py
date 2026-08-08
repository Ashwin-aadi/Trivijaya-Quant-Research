from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion20d(Strategy):
    rationale = (
        "Price reverts to the mean over time. By identifying assets that are far from their "
        "trailing average, we can identify potential reversions and profit from such movements."
    )

    def __init__(self, window: int = 20, zscore_threshold: float = 1.5) -> None:
        self._window = window
        self._zscore_threshold = zscore_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in closes.columns]
        means = closes.select(pl.col(symbols).mean()).to_dict(True)
        zscores = closes.with_columns(
            (pl.col(symbols) - pl.col(symbols).shift(self._window)).over(symbols) / (
                pl.col(symbols).std().over(symbols) * self._window ** 0.5
            ).alias(f"zscore_{self._window}")
        )
        zscores = {symbol: float(zs[0]) for symbol, zs in zscores.to_dict(True).items()}

        filtered_symbols = [symbol for symbol, score in zscores.items() if abs(score) > self._zscore_threshold]
        weight = 1.0 / len(filtered_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in filtered_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest