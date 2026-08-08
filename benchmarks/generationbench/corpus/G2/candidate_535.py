from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression suggests that when a stock's price volatility decreases significantly "
        "over a period, it often leads to a breakout in the near future. This phenomenon can be used "
        "to identify potential trading opportunities."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        range_compression_scores = {}
        for symbol in view.symbols:
            if symbol not in history["symbol"].unique().to_list():
                continue
            highs = history.filter(pl.col("symbol") == symbol)["high"].to_list()
            lows = history.filter(pl.col("symbol") == symbol)["low"].to_list()

            high_diffs = [highs[i] - max(highs[:i]) for i in range(1, len(highs))]
            low_diffs = [lows[i] - min(lows[:i]) for i in range(1, len(lows))]

            if not all(high_diffs) or not all(low_diffs):
                continue

            max_highs = max(high_diffs)
            min_lows = min(low_diffs)

            range_compression_score = (min_lows + max_highs) / 2
            range_compression_scores[symbol] = range_compression_score

        if not range_compression_scores:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = sorted(range_compression_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s, _ in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest