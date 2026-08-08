from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression20d(Strategy):
    rationale = (
        "Range compression occurs when the high and low of a stock are closer to each other "
        "than their historical average. This suggests increased price volatility in the near future."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbol_range_compression = {}

        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            high = float(history[symbol].filter(pl.col("session_date") == stamp).get(1))
            low = float(history[symbol].filter(pl.col("session_date") == stamp).get(2))
            recent_highs_lowss = (
                history.with_columns(
                    pl.col("high").shift_and_fill(self._window - 1, high),
                    pl.col("low").shift_and_fill(self._window - 1, low),
                )
                .select(
                    "symbol",
                    (pl.col("close") - pl.col("high")).abs().max() + (pl.col("close") - pl.col("low")).abs().max(),
                )
                .group_by("symbol")
                .agg(pl.col(0).mean())
            ).to_dict(as_pandas=False)

            if recent_highs_lowss:
                symbol_range_compression[symbol] = recent_highs_lowss[stamp][0]

        sorted_symbols = [
            s for s, v in sorted(symbol_range_compression.items(), key=lambda item: item[1], reverse=True)
        ][:5]
        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in sorted_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest