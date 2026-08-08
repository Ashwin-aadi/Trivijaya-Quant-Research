from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression occurs when a stock's price movement becomes less volatile over "
        "a period. This can indicate that the market is consolidating and may be setting up for "
        "a potential breakout or trend continuation. Identifying stocks with reduced volatility "
        "can help predict such movements."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        ranges = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            highs = [float(h) for h in history[symbol + "_high"].to_list()]
            lows = [float(l) for l in history[symbol + "_low"].to_list()]
            range_series = pl.Series(highs).sub(pl.Series(lows)).alias("range")
            range_value = range_series.sum()
            ranges.append((symbol, range_value))

        if not ranges:
            return Signal(information_available_at=stamp, weights={})

        avg_range = sum(r[1] for r in ranges) / len(ranges)
        compressed_stocks = [r[0] for r in ranges if r[1] < 0.75 * avg_range]

        weight = 1.0 / len(compressed_stocks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in compressed_stocks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest