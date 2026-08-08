from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Stocks with compressed ranges may indicate a lack of directionality and can be "
        "potential candidates for breakout or mean reversion strategies. By identifying "
        "such stocks, we aim to capture the potential for price movement."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        range_compression = {
            symbol: float(history[("session_date", "close")].group_by("symbol").agg(
                (pl.col("close").max() - pl.col("close").min()).alias("range")
            ).sort("range").first()["range"])
            for symbol in symbols
        }

        sorted_symbols = [k for k, v in sorted(range_compression.items(), key=lambda item: item[1])]
        if not sorted_symbols:
            return Signal(information_available_at=stamp, weights={})

        top_n_symbols = sorted_symbols[:5]
        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_n_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        raise ValueError("No historical data available.")
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest