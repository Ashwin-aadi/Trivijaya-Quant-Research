from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression is a condition where the price action within a trading period "
        "becomes more concentrated. This can indicate increased volatility and potential "
        "market turning points, making it a valuable signal for entering trades."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        range_compression_scores = []
        for symbol in symbols:
            opens = history[f"{symbol}_open"].to_list()
            closes = history[f"{symbol}_close"].to_list()

            max_price = max(opens + closes)
            min_price = min(opens + closes)

            range_score = (max_price - min_price) / sum(closes)
            range_compression_scores.append((symbol, range_score))

        range_compression_scores.sort(key=lambda x: x[1], reverse=True)
        top_symbols = [score[0] for score in range_compression_scores[:5]]

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
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest