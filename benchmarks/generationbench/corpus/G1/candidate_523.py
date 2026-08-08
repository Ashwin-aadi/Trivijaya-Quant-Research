from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression signals periods of reduced volatility, which often precede "
        "price movements. Identifying such periods can provide opportunities for profit."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        range_compression_scores = {}
        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue
            df = history.filter(pl.col("symbol") == symbol)
            opens = [float(o) for o in df.select("open").to_series().drop_nulls().to_list()]
            closes = [float(c) for c in df.select("close").to_series().drop_nulls().to_list()]
            if len(opens) < self._window or len(closes) < self._window:
                continue

            high_range = max(opens + closes)
            low_range = min(opens + closes)
            range_compression_score = (high_range - low_range) / sum(abs(open_val - close_val) for open_val, close_val in zip(opens[:-1], closes[1:]))
            range_compression_scores[symbol] = range_compression_score

        top_symbols = sorted(range_compression_scores, key=range_compression_scores.get, reverse=True)[:5]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select("session_date").max().to_series()[0]
    assert isinstance(newest, date)
    return newest