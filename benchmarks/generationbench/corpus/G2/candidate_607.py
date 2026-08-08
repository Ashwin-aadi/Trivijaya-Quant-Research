from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompressionStrategy(Strategy):
    rationale = (
        "Range compression occurs when volatility decreases over a period. This can indicate "
        "a potential for increased price movement in the future. By identifying symbols with "
        "recent range compression, we aim to capture this opportunity."
    )

    def __init__(self, window: int = 20, threshold: float = 0.95) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        range_compression_scores = {}
        for symbol in view.symbols:
            open_values = [float(v) for v in history[symbol]["open"].to_list()]
            close_values = [float(v) for v in history[symbol]["close"].to_list()]

            high_low_diff = max([high - low for high, low in zip(history[symbol]["high"], history[symbol]["low"])])

            if high_low_diff == 0:
                range_compression_scores[symbol] = 1.0
                continue

            open_close_diff = abs(open_values[-1] - close_values[-1])
            compression_score = open_close_diff / high_low_diff

            if compression_score < self._threshold:
                range_compression_scores[symbol] = compression_score

        # Sort symbols by their range compression score from highest to lowest
        sorted_symbols = [s for s, c in sorted(range_compression_scores.items(), key=lambda item: -item[1])]
        
        top_n_symbols = sorted_symbols[:3]

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