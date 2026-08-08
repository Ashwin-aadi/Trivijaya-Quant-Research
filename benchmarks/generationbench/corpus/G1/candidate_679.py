from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression indicates that the price action is consolidating and may be "
        "about to break out. This strategy identifies symbols with reduced volatility over a "
        "recent period."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        range_compression_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            close_values = [float(v) for v in history[symbol].to_list()]
            if len(close_values) < self._window:
                continue

            high_low_range = pl.DataFrame(
                {"high": close_values, "low": close_values}
            ).select(
                (pl.col("high") - pl.col("low")).alias("range")
            ).height
            average_range = (
                pl.DataFrame({"close": close_values})
                .select((pl.col("close").max() - pl.col("close").min()).alias("total"))
                .item()
                / self._window
            )
            score = high_low_range / average_range if average_range > 0 else 0.0
            range_compression_scores[symbol] = score

        sorted_symbols = [
            s for s, _ in sorted(range_compression_scores.items(), key=lambda item: -item[1])
        ]
        top_n_symbols = sorted_symbols[:5]
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