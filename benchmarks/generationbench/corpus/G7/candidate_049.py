from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion5d(Strategy):
    rationale = (
        "Mean reversion occurs when an asset's price moves towards its historical average. "
        "This strategy aims to capture this phenomenon by identifying stocks that have "
        "deviated significantly from their mean over the past 5 days and then going long on them."
    )

    def __init__(self, window: int = 5, top_n: int = 30) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window * 2 + 1:
            return Signal(information_available_at=stamp, weights={})

        mean_close = (
            closes.group_by("symbol").agg(
                (pl.col("adj_close") - pl.col("adj_close").shift(self._window)).mean().alias("deviation")
            )
        )["deviation"].to_list()[0]

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window * 2 + 1:
                continue
            deviation = (values[-1] - values[self._window]) / values[self._window]
            if abs(deviation) > mean_close:
                picks.append(symbol)

        picks = picks[: self._top_n]
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