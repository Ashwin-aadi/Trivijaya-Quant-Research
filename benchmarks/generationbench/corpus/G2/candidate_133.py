from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversionShortHorizon(Strategy):
    rationale = (
        "Mean reversion strategies exploit the tendency for stock prices to revert to their mean "
        "over short periods. If a stock has deviated significantly from its historical average, "
        "it is likely to move back towards that average. This can be measured by calculating the "
        "deviation of recent close prices from a simple moving average."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_close = closes.select(pl.col("adj_close").mean().alias("mean")).to_dict(
            full_series=True
        )["mean"][0]
        deviations = {
            symbol: float(close - mean_close) for symbol, close in zip(view.symbols, closes.row(0))
        }

        sorted_symbols = [
            s for s, d in sorted(deviations.items(), key=lambda item: abs(item[1]), reverse=True)
        ][:5]

        if not sorted_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: -weight for s in sorted_symbols},  # Short the most deviated symbols
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest