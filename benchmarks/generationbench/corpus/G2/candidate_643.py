from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression occurs when the high-low range of stock prices decreases over time. "
        "This can indicate that the market is consolidating and may be setting up for a breakout. "
        "High compression often precedes price action that could lead to a significant move."
    )

    def __init__(self, lookback_days: int = 20) -> None:
        self._lookback_days = lookback_days

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_days)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        range_values = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            data = [float(v) for v in history[symbol].to_list()]
            high_low_range = (max(data) - min(data)) / max(data)
            range_values.append((symbol, high_low_range))

        range_df = pl.DataFrame(range_values).with_columns(
            (pl.col(1) / pl.col(1).shift(1) - 1.0).alias("range_change")
        ).sort("range_change", descending=True)

        if range_df.height < 5:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = [row[0] for row in range_df.to_numpy()[:5]]
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