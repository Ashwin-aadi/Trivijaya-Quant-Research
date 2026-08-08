from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression indicates that the stock prices are consolidating within a "
        "narrower range than usual. This can be an indication of potential breakout or consolidation "
        "patterns, making it favorable for entry into positions."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        range_compression_score = {}
        for symbol in view.symbols:
            history_symbol = history.select(
                pl.col("session_date"), pl.col(symbol).alias("close")
            )
            if history_symbol.height < self._window + 1:
                continue

            recent_highs = (
                history_symbol.sort("session_date", descending=False)
                .select(pl.col("close").max().alias("recent_high"))
                .to_series()
            )
            recent_lows = (
                history_symbol.select(pl.col("close").min().alias("recent_low"))
                .to_series()
            )

            high_diffs = (recent_highs[1:] - recent_highs[:-1]).to_list()[::-1]
            low_diffs = (recent_lows[1:] - recent_lows[:-1]).to_list()[::-1]

            mean_range = sum(high_diffs) / len(high_diffs)
            std_dev_range = (sum((x - mean_range) ** 2 for x in high_diffs + low_diffs) / len(
                high_diffs
            )) ** 0.5

            range_compression_score[symbol] = (
                sum(abs(x) for x in high_diffs + low_diffs) / (std_dev_range * self._window)
            )

        sorted_scores = sorted(range_compression_score.items(), key=lambda x: -x[1])
        if not sorted_scores:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = [symbol for symbol, _ in sorted_scores[:5]]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest