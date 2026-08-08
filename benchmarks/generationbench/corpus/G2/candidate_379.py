from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompressionStrategy(Strategy):
    rationale = (
        "Range compression often signals that liquidity is being accumulated by market "
        "makers or institutional players. This can lead to a breakout in the near future. "
        "By identifying symbols with increased price range, we aim to capture this potential "
        "momentum shift."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols_with_range = []
        for symbol in view.symbols:
            if symbol not in history["symbol"].unique().to_list():
                continue
            daily_highs = (
                history.select(pl.col("high").filter(pl.col("symbol") == symbol))
                .sort(by="session_date")
                .select(pl.col("high"))
                .to_numpy()
                .flatten()
            )
            daily_lows = (
                history.select(pl.col("low").filter(pl.col("symbol") == symbol))
                .sort(by="session_date")
                .select(pl.col("low"))
                .to_numpy()
                .flatten()
            )
            if len(daily_highs) < self._window or len(daily_lows) < self._window:
                continue
            current_range = daily_highs[-1] - daily_lows[-1]
            past_ranges = [daily_high - low for high, low in zip(daily_highs[1:], daily_lows)]
            avg_past_range = sum(past_ranges) / len(past_ranges)
            if current_range < 0.9 * avg_past_range:
                symbols_with_range.append(symbol)

        top_symbols = sorted(symbols_with_range, key=lambda s: -view.closes(lookback=None)[s][-1])[:5]
        weight = 1.0 / len(top_symbols) if top_symbols else 0
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest