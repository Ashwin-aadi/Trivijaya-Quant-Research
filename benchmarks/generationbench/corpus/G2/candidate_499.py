from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompressionStrategy(Strategy):
    rationale = (
        "Range compression occurs when a stock's price movement becomes more concentrated "
        "within a narrower range over time. This can indicate increased market inefficiency or "
        "sentiment-driven volatility, which may provide opportunities for momentum-based trading."
    )

    def __init__(self, window: int = 60) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        range_compression_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue
            high_low_range = max(values) - min(values)
            recent_high = float(max(values[-10:]))
            recent_low = float(min(values[-10:]))
            compression_score = (recent_high - recent_low) / high_low_range
            range_compression_scores[symbol] = compression_score

        sorted_symbols = [
            s for _, s in sorted(range_compression_scores.items(), key=lambda item: item[1])
        ]
        top_n_symbols = sorted_symbols[-5:]
        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_n_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest