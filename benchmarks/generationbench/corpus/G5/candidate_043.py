from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "During periods of range compression, the volatility tends to increase in subsequent "
        "periods. By identifying such periods, we can potentially capture profitable trades when "
        "the market breaks out of its tight range."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        range_compression_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.symbol.unique().to_list():
                continue
            opens = history.select(pl.col("open")[history["symbol"] == symbol])
            closes = history.select(pl.col("close")[history["symbol"] == symbol])
            opens_values = [float(v) for v in opens.drop_nulls().to_list()[0]]
            closes_values = [float(v) for v in closes.drop_nulls().to_list()[0]]

            if len(opens_values) < self._window or len(closes_values) < self._window:
                continue

            high_range = max(opens_values[-self._window:]) - min(opens_values[-self._window:])
            low_range = max(min(closes_values[-self._window:]) - closes_values[:-1])

            if high_range > 0 and low_range > 0:
                score = (high_range + low_range) / sum(
                    [abs(o - c) for o, c in zip(opens_values[-self._window:], closes_values[-self._window-1:-1])]
                )
                range_compression_scores[symbol] = score

        if not range_compression_scores:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = sorted(range_compression_scores.items(), key=lambda item: item[1], reverse=True)[:5]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s, _ in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().cast(pl.Date)
    assert isinstance(newest, date)
    return newest