from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion20d(Strategy):
    rationale = (
        "Reversion to the mean suggests that prices which have moved significantly away "
        "from their average level over a period will likely return. This strategy exploits "
        "this tendency by selling stocks that are far from their mean and buying those that "
        "are close, based on their recent price levels."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)

        if closes.height < self._window or len(closes.columns) <= 1:
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in closes.columns]
        mean_close = (
            closes.select(pl.col(symbols).mean().alias("mean"))
            .with_columns(
                (pl.col(symbols) - pl.col("mean")).abs().rank(method="dense", descending=False).alias("deviation_rank")
            )
            .select("deviation_rank")
            .to_series()
        )

        sorted_mean_close = mean_close.to_list()

        weights: dict[str, float] = {}
        for symbol in symbols:
            rank = sorted_mean_close.index(symbol)
            if rank < self._window // 4 or rank > 3 * self._window // 4:
                continue
            weight = (1.0 / len(symbols)) * ((self._window - rank) / self._window)
            weights[symbol] = weight

        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest